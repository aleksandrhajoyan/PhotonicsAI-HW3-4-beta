"""Acquire exactly one frame from the iRAYPLE MV-A7200MU130 via the official HuaRay IMV SDK.

Manual, single-shot utility. Nothing is loaded from the vendor SDK on import.
main() touches the camera only with --capture-once and an interactive typed
confirmation. One IMV_GetFrame call is made with a bounded timeout and is never
retried. No camera feature is written: exposure, gain, pixel format, frame
rate, ROI and trigger mode are left exactly as the camera reports them. The one
feature access is a read of TriggerMode, used to refuse (not reconfigure) a
camera that is waiting for a trigger.

SDK facts this module relies on (Development/Include/IMV/IMVApi.h, IMVDefines.h):
    IMV_CreateHandle(OUT IMV_HANDLE*, IN IMV_ECreateHandleMode, IN void* pIdentifier)
        modeByCameraKey takes a char* camera key ("vendor:serial").
    IMV_GetFrame(IN IMV_HANDLE, OUT IMV_Frame*, IN unsigned int timeoutMS)
        Uses an SDK-internal buffer that must be returned with IMV_ReleaseFrame.
    IMV_FrameInfo.status: 0 is the normal frame status.
    IMV_Frame.pData: start of the image data; frameInfo.size: image size in bytes.
    All calls return IMV_OK (0) on success, a negative error code otherwise.
"""

import argparse
import ctypes
import hashlib
import sys
from dataclasses import dataclass, field

import irayple_discovery as discovery
from camera_contract import SOURCE_HARDWARE

EXPECTED_MODEL = discovery.EXPECTED_MODEL
CONFIRMATION_PHRASE = "CAPTURE ONE FRAME"
GET_FRAME_TIMEOUT_MS = 3000
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_DIMENSION = 16384
MAX_ENUMERATED_DEVICES = 100  # IMV_MAX_DEVICE_ENUM_NUM

EXIT_OK = 0
EXIT_CAPTURE_FAILED = 1
EXIT_REFUSED = 2
EXIT_DEVICE_SELECTION = 3
EXIT_CLEANUP_FAILED = 4
EXIT_INTERRUPTED = 130

# Error codes defined in the C header IMVDefines.h but absent from the Python IMVDefines.py.
IMV_TIMEOUT = -119
HEADER_ONLY_ERROR_NAMES = {
    -114: "IMV_RESTORE_STREAM",
    -115: "IMV_RECONNECT_DEVICE",
    -116: "IMV_NOT_AVAILABLE",
    -117: "IMV_NOT_GRABBING",
    -118: "IMV_NOT_CONNECTED",
    IMV_TIMEOUT: "IMV_TIMEOUT",
    -120: "IMV_IS_CONNECTED",
    -121: "IMV_IS_GRABBING",
    -122: "IMV_INVOCATION_ERROR",
    -123: "IMV_SYSTEM_ERROR",
    -124: "IMV_OPENFILE_ERROR",
}

# IMV_EPixelType encodes bits occupied per pixel in bits 16-23 (IMV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_MASK).
PIXEL_SIZE_MASK = 0x00FF0000
PIXEL_SIZE_SHIFT = 16

STATUS_CAPTURED = "RAW_FRAME_CAPTURED"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
DECODE_MONO8 = "MONO8_STATS"
DECODE_UNAVAILABLE = "PIXEL_DECODE_UNAVAILABLE"

DEVICE_SELECTION_STATUSES = ("NO_DEVICE", "MULTIPLE_DEVICES", "UNEXPECTED_DEVICE", "UNEXPECTED_TRANSPORT")

SCOPE_NOTE = (
    "Single uncalibrated frame. No claim is made about illumination, sensor sensitivity, "
    "optical focus, temporal noise or physical units. Frame bytes are held in memory only "
    "and discarded when this program exits."
)


class CaptureError(Exception):
    """Controlled acquisition failure, tagged with a status and optional native code."""

    def __init__(self, status, message, sdk_code=None):
        super().__init__(message)
        self.status = status
        self.sdk_code = sdk_code


@dataclass(frozen=True)
class SelectedDevice:
    model: str
    masked_serial: str
    camera_key: bytes


@dataclass(frozen=True)
class CleanupStep:
    name: str
    code: object = None
    error: str = None

    @property
    def ok(self):
        return self.error is None and self.code == 0


@dataclass(frozen=True)
class CapturedFrame:
    model: str
    frame_id: int
    width: int
    height: int
    pixel_format: int
    pixel_format_label: str
    padding_x: int
    padding_y: int
    payload_bytes: int
    data: bytes
    sha256: str
    acquisition_source: str
    decode_status: str
    decode_note: str
    pixel_min: int = None
    pixel_max: int = None
    pixel_mean: float = None


@dataclass
class CaptureResult:
    status: str
    message: str = ""
    sdk_code: int = None
    sdk_code_name: str = None
    device: SelectedDevice = None
    trigger_mode: str = None
    frame: CapturedFrame = None
    cleanup: list = field(default_factory=list)

    @property
    def cleanup_ok(self):
        return all(step.ok for step in self.cleanup)

    @property
    def ok(self):
        return self.status == STATUS_CAPTURED and self.frame is not None and self.cleanup_ok


def error_label(sdk, code):
    name = discovery.error_name(sdk, code)
    if name == "unrecognized":
        name = HEADER_ONLY_ERROR_NAMES.get(code, name)
    return name


def pixel_format_label(sdk, pixel_format):
    for name, value in vars(sdk.IMV_EPixelType).items():
        if name.startswith("gvspPixel") and value == pixel_format:
            return name[len("gvspPixel"):]
    return None


def bits_per_pixel(pixel_format):
    if pixel_format <= 0:
        return None
    bits = (pixel_format & PIXEL_SIZE_MASK) >> PIXEL_SIZE_SHIFT
    return bits or None


def select_device(sdk):
    """Enumerate USB3 cameras once; require exactly one device and that it is the expected model."""
    device_list = sdk.IMV_DeviceList()
    code = sdk.MvCamera.IMV_EnumDevices(device_list, sdk.IMV_EInterfaceType.interfaceTypeUsb3)
    if code != sdk.IMV_OK:
        raise CaptureError("ENUMERATION_FAILED", "IMV_EnumDevices failed", sdk_code=code)
    count = int(device_list.nDevNum)
    if count > MAX_ENUMERATED_DEVICES:
        raise CaptureError("ENUMERATION_FAILED", f"implausible device count {count}")
    if count == 0:
        raise CaptureError("NO_DEVICE", "the SDK reported no USB3 Vision cameras")
    if count > 1:
        raise CaptureError("MULTIPLE_DEVICES", f"{count} USB3 cameras present; refusing to choose one")

    info = device_list.pDevInfo[0]
    record = discovery.parse_device(0, info)
    if not discovery.is_expected(record):
        raise CaptureError("UNEXPECTED_DEVICE", f"found model {record.model!r}, expected {EXPECTED_MODEL}")
    if record.camera_type != sdk.typeU3vCamera or record.interface_type != sdk.IMV_EInterfaceType.interfaceTypeUsb3:
        raise CaptureError(
            "UNEXPECTED_TRANSPORT",
            f"nCameraType={record.camera_type}, nInterfaceType={record.interface_type}; expected USB3 Vision",
        )
    camera_key = info.cameraKey
    if not isinstance(camera_key, bytes) or not camera_key:
        raise CaptureError("UNEXPECTED_DEVICE", "device reported no camera key")
    return SelectedDevice(
        model=record.model,
        masked_serial=discovery.mask_identifier(record.serial_number),
        camera_key=camera_key,
    )


def validate_frame_metadata(sdk, frame):
    """Read and bound-check frame metadata. Returns plain ints; reads no pixel memory."""
    info = frame.frameInfo
    status = int(info.status)
    width = int(info.width)
    height = int(info.height)
    size = int(info.size)
    pixel_format = int(info.pixelFormat)
    if status != 0:
        raise CaptureError("INVALID_FRAME", f"frame status {status} is not the normal status 0")
    if not frame.pData:
        raise CaptureError("INVALID_FRAME", "frame data pointer is NULL")
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise CaptureError("INVALID_FRAME", f"implausible frame dimensions {width}x{height}")
    if size <= 0:
        raise CaptureError("INVALID_FRAME", f"invalid payload size {size}")
    if size > MAX_PAYLOAD_BYTES:
        raise CaptureError("INVALID_FRAME", f"payload size {size} exceeds the {MAX_PAYLOAD_BYTES}-byte limit")
    label = pixel_format_label(sdk, pixel_format)
    bits = bits_per_pixel(pixel_format) if label is not None else None
    if bits is not None:
        required = (width * height * bits + 7) // 8
        if size < required:
            raise CaptureError(
                "INVALID_FRAME",
                f"payload size {size} is smaller than {required} bytes required for {width}x{height} {label}",
            )
    return {
        "frame_id": int(info.blockId),
        "width": width,
        "height": height,
        "size": size,
        "pixel_format": pixel_format,
        "label": label,
        "padding_x": int(info.paddingX),
        "padding_y": int(info.paddingY),
    }


def mono8_statistics(sdk, meta, data):
    """Min/max/mean only for positively identified, unpadded, exactly sized Mono8."""
    if meta["pixel_format"] != sdk.IMV_EPixelType.gvspPixelMono8:
        return DECODE_UNAVAILABLE, "pixel format is not Mono8; raw bytes not interpreted", None
    if meta["padding_x"] != 0 or meta["padding_y"] != 0:
        return DECODE_UNAVAILABLE, "Mono8 frame reports padding; row layout not verified", None
    if len(data) != meta["width"] * meta["height"]:
        return DECODE_UNAVAILABLE, "Mono8 payload size does not equal width*height", None
    return DECODE_MONO8, "Mono8, unpadded, size == width*height", (min(data), max(data), sum(data) / len(data))


class _Session:
    """Tracks which native resources were acquired so cleanup releases exactly those, in reverse order."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.camera = None
        self.handle_created = False
        self.opened = False
        self.grabbing = False
        self.frame = None
        self.frame_acquired = False
        self.frame_release_attempted = False
        self.interrupted = False
        self.steps = []

    def _check(self, status, name, code):
        if code != self.sdk.IMV_OK:
            raise CaptureError(status, f"{name} failed", sdk_code=code)

    def read_trigger_mode(self):
        symbol = self.sdk.IMV_String()
        code = self.camera.IMV_GetEnumFeatureSymbol("TriggerMode", symbol)
        if code != self.sdk.IMV_OK:
            return f"unknown (read failed with code {code} {error_label(self.sdk, code)})"
        return discovery.decode_field(symbol.str)

    def run(self, result):
        device = select_device(self.sdk)
        result.device = device

        self.camera = self.sdk.MvCamera()
        camera_key = ctypes.c_char_p(device.camera_key)
        code = self.camera.IMV_CreateHandle(self.sdk.IMV_ECreateHandleMode.modeByCameraKey, camera_key)
        self._check("CREATE_HANDLE_FAILED", "IMV_CreateHandle", code)
        self.handle_created = True

        self._check("OPEN_FAILED", "IMV_Open", self.camera.IMV_Open())
        self.opened = True

        result.trigger_mode = self.read_trigger_mode()
        if result.trigger_mode == "On":
            raise CaptureError("TRIGGER_MODE_ON", "camera TriggerMode is On; trigger settings left unchanged")

        self._check("START_GRABBING_FAILED", "IMV_StartGrabbing", self.camera.IMV_StartGrabbing())
        self.grabbing = True

        self.frame = self.sdk.IMV_Frame()
        code = self.camera.IMV_GetFrame(self.frame, GET_FRAME_TIMEOUT_MS)
        self._check("GET_FRAME_FAILED", f"IMV_GetFrame (timeout {GET_FRAME_TIMEOUT_MS} ms)", code)
        self.frame_acquired = True

        meta = validate_frame_metadata(self.sdk, self.frame)
        data = ctypes.string_at(self.frame.pData, meta["size"])
        self.release_frame()

        decode_status, decode_note, stats = mono8_statistics(self.sdk, meta, data)
        pixel_min, pixel_max, pixel_mean = stats if stats else (None, None, None)
        result.frame = CapturedFrame(
            model=device.model,
            frame_id=meta["frame_id"],
            width=meta["width"],
            height=meta["height"],
            pixel_format=meta["pixel_format"],
            pixel_format_label=meta["label"] or "unrecognized",
            padding_x=meta["padding_x"],
            padding_y=meta["padding_y"],
            payload_bytes=len(data),
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            acquisition_source=SOURCE_HARDWARE,
            decode_status=decode_status,
            decode_note=decode_note,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
            pixel_mean=pixel_mean,
        )
        result.status = STATUS_CAPTURED

    def _cleanup_call(self, name, call):
        try:
            code = call()
        except KeyboardInterrupt:
            self.interrupted = True
            self.steps.append(CleanupStep(name, error="interrupted by KeyboardInterrupt"))
        except Exception as exc:  # native wrapper failure must not skip the remaining cleanup
            self.steps.append(CleanupStep(name, error=repr(exc)))
        else:
            self.steps.append(CleanupStep(name, code=code))

    def release_frame(self):
        if not self.frame_acquired or self.frame_release_attempted:
            return
        self.frame_release_attempted = True
        frame, self.frame = self.frame, None
        self._cleanup_call("IMV_ReleaseFrame", lambda: self.camera.IMV_ReleaseFrame(frame))

    def cleanup(self):
        self.release_frame()
        if self.grabbing:
            self.grabbing = False
            self._cleanup_call("IMV_StopGrabbing", self.camera.IMV_StopGrabbing)
        if self.opened:
            self.opened = False
            self._cleanup_call("IMV_Close", self.camera.IMV_Close)
        if self.handle_created:
            self.handle_created = False
            self._cleanup_call("IMV_DestroyHandle", self.camera.IMV_DestroyHandle)
        return list(self.steps)


def capture_one_frame(sdk):
    """Run one enumerate/open/grab/GetFrame/release/close cycle. Never raises for SDK failures.

    Only a KeyboardInterrupt that escapes cleanup itself can propagate.
    """
    session = _Session(sdk)
    result = CaptureResult(status=STATUS_UNEXPECTED_ERROR)
    try:
        session.run(result)
    except CaptureError as exc:
        result.status = exc.status
        result.message = str(exc)
        result.sdk_code = exc.sdk_code
        if exc.sdk_code is not None:
            result.sdk_code_name = error_label(sdk, exc.sdk_code)
    except KeyboardInterrupt:
        result.status = STATUS_INTERRUPTED
        result.message = "interrupted before acquisition completed"
    except Exception as exc:
        result.status = STATUS_UNEXPECTED_ERROR
        result.message = repr(exc)
    finally:
        result.cleanup = session.cleanup()
    if session.interrupted:
        result.status = STATUS_INTERRUPTED
        result.message = result.message or "interrupted during cleanup"
    return result


def format_result(result):
    lines = []
    if result.device is not None:
        lines.append(f"Model:              {result.device.model}")
        lines.append(f"Serial:             {result.device.masked_serial}")
    if result.trigger_mode is not None:
        lines.append(f"TriggerMode (read): {result.trigger_mode}")
    frame = result.frame
    if frame is not None:
        lines.extend(
            [
                f"Frame ID (blockId): {frame.frame_id}",
                f"Width x Height:     {frame.width} x {frame.height}",
                f"Pixel format:       0x{frame.pixel_format & 0xFFFFFFFF:08X} ({frame.pixel_format_label})",
                f"Padding X / Y:      {frame.padding_x} / {frame.padding_y}",
                f"Payload bytes:      {frame.payload_bytes}",
                f"Acquisition source: {frame.acquisition_source}",
                f"SHA-256:            {frame.sha256}",
            ]
        )
        if frame.decode_status == DECODE_MONO8:
            lines.append(
                f"Pixel decode:       {DECODE_MONO8} min={frame.pixel_min} max={frame.pixel_max} "
                f"mean={frame.pixel_mean:.3f}"
            )
        else:
            lines.append(f"Pixel decode:       {DECODE_UNAVAILABLE} ({frame.decode_note})")
    for step in result.cleanup:
        if step.ok:
            lines.append(f"Cleanup {step.name}: OK")
        elif step.error is not None:
            lines.append(f"Cleanup {step.name}: FAILED ({step.error})")
        else:
            lines.append(f"Cleanup {step.name}: FAILED with code {step.code}")

    if result.ok:
        lines.append(f"CAPTURE_STATUS: {STATUS_CAPTURED}")
        lines.append("FRAME_RELEASED_AND_CAMERA_SHUT_DOWN")
    elif result.status == STATUS_CAPTURED:
        lines.append(f"CAPTURE_STATUS: {STATUS_CAPTURED}")
        lines.append("CLEANUP_FAILURE: one or more SDK release/shutdown calls failed (see above)")
    else:
        detail = result.message
        if result.sdk_code is not None:
            detail += f" (error code {result.sdk_code} {result.sdk_code_name})"
        lines.append(f"CAPTURE_FAILED: {result.status}: {detail}")
        if not result.cleanup_ok:
            lines.append("CLEANUP_FAILURE: one or more SDK release/shutdown calls failed (see above)")
    return lines


def exit_code_for(result):
    if result.ok:
        return EXIT_OK
    if result.status == STATUS_INTERRUPTED:
        return EXIT_INTERRUPTED
    if result.status == STATUS_CAPTURED:
        return EXIT_CLEANUP_FAILED
    if result.status in DEVICE_SELECTION_STATUSES:
        return EXIT_DEVICE_SELECTION
    return EXIT_CAPTURE_FAILED


def _is_interactive(stream):
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def main(argv=None, sdk_loader=None, input_fn=input, stdin=None):
    parser = argparse.ArgumentParser(
        description="Acquire exactly one frame from the iRAYPLE MV-A7200MU130 via the official HuaRay IMV SDK."
    )
    parser.add_argument(
        "--capture-once",
        action="store_true",
        help="required: open the camera and request one frame (interactive confirmation follows)",
    )
    args = parser.parse_args(argv)

    print("LUMEN iRAYPLE first frame (single manual acquisition)")
    if not args.capture_once:
        print("NO_ACTION: pass --capture-once to request a single frame. No camera was accessed.")
        return EXIT_REFUSED

    stdin = sys.stdin if stdin is None else stdin
    if not _is_interactive(stdin):
        print("REFUSED: an interactive terminal is required for confirmation. No camera was accessed.")
        return EXIT_REFUSED
    try:
        response = input_fn(f"Type {CONFIRMATION_PHRASE} to open the camera and acquire one frame: ")
    except (EOFError, KeyboardInterrupt):
        print("\nREFUSED: no confirmation received. No camera was accessed.")
        return EXIT_REFUSED
    if response != CONFIRMATION_PHRASE:
        print("REFUSED: confirmation phrase did not match. No camera was accessed.")
        return EXIT_REFUSED

    loader = sdk_loader if sdk_loader is not None else discovery.load_vendor_sdk
    try:
        sdk = loader()
    except discovery.DiscoveryError as exc:
        print(f"CAPTURE_FAILED: SDK_LOAD_FAILED: {exc}")
        return EXIT_CAPTURE_FAILED

    try:
        result = capture_one_frame(sdk)
    except KeyboardInterrupt:
        print("CAPTURE_FAILED: INTERRUPTED during cleanup; camera state unknown.")
        return EXIT_INTERRUPTED
    for line in format_result(result):
        print(line)
    if result.frame is not None:
        print(SCOPE_NOTE)
    return exit_code_for(result)


if __name__ == "__main__":
    sys.exit(main())
