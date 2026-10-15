"""Controlled two-frame ExposureTime sweep on the iRAYPLE MV-A7200MU130 via the official HuaRay IMV SDK.

Manual, single-session experiment. Nothing is loaded from the vendor SDK on import.
main() touches the camera only with --sweep-once, --unit-us-verified, an
interactive terminal and the typed confirmation phrase. One camera session:
enumerate, CreateHandle, Open, verify preconditions (TriggerMode, ExposureAuto,
GainAuto all Off; ExposureTime == 5000.0 within [min, max]), then for each of
1000.0 and 5000.0: write ExposureTime, read it back, StartGrabbing, one
IMV_GetFrame, copy, ReleaseFrame, StopGrabbing. The originally read
ExposureTime is always restored and read back before Close and DestroyHandle
whenever any exposure write was issued. Frames are never retried. The only
feature ever written is ExposureTime.

SDK facts this module relies on (Development/Include/IMV/IMVApi.h,
Samples/Python/IMV/MVSDK/IMVApi.py, Samples/Python/IMV/CommPropAccess):
    int IMV_SetDoubleFeatureValue(IN IMV_HANDLE, IN const char* pFeatureName, IN double doubleValue)
        Returns IMV_OK (0) on success, an error code otherwise. The Python wrapper
        takes (str, float), encodes the name as UTF-8 and wraps the value in
        ctypes.c_double itself. The vendor CommPropAccess sample writes
        ExposureTime after IMV_Open, before grabbing, and reads it back with
        IMV_GetDoubleFeatureValue to observe the applied value.
    bool IMV_FeatureIsWriteable(IN IMV_HANDLE, IN const char* pFeatureName)
    IMV_GetFrame uses an SDK-internal buffer that must be returned with IMV_ReleaseFrame.

ExposureTime unit: the installed SDK headers and bindings do not state it. The
unit basis used here is the human operator's check in MV Viewer, asserted with
--unit-us-verified (microseconds).
"""

import argparse
import ctypes
import math
import sys
from dataclasses import dataclass, field

import numpy as np

import irayple_discovery as discovery
import irayple_exposure_info as exposure_info
import irayple_first_frame as first_frame
from camera_contract import SOURCE_HARDWARE
from measurements import analyze_frame

EXPECTED_MODEL = discovery.EXPECTED_MODEL
CONFIRMATION_PHRASE = "RUN EXPOSURE SWEEP"
EXPOSURE_FEATURE = exposure_info.EXPOSURE_FEATURE
REQUIRED_OFF_FEATURES = ("TriggerMode", "ExposureAuto", "GainAuto")
EXPECTED_ORIGINAL_EXPOSURE = 5000.0
PLANNED_EXPOSURES = (1000.0, 5000.0)
# 1000.0 and 5000.0 are exactly representable in IEEE-754 binary32 and binary64,
# so a device storing them in either format reads them back bit-exact. The
# tolerance only absorbs representation round-off; any device-side quantization
# that actually changes the exposure is reported as a mismatch, never accepted.
EXPOSURE_TOLERANCE = 1e-6
UNIT_BASIS = (
    "microseconds (us), human-verified in MV Viewer (--unit-us-verified); "
    "not stated by the installed IMV SDK headers"
)
MONO8_BIT_DEPTH = 8

EXIT_OK = 0
EXIT_SWEEP_FAILED = 1
EXIT_REFUSED = 2
EXIT_DEVICE_SELECTION = 3
EXIT_CLEANUP_FAILED = 4
EXIT_PRECONDITION = 5
EXIT_RESTORE_FAILED = 6
EXIT_INTERRUPTED = 130

STATUS_COMPLETE = "EXPOSURE_SWEEP_COMPLETE"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
PRECONDITION_STATUSES = (
    "FEATURE_UNAVAILABLE", "TRIGGER_MODE_NOT_OFF", "EXPOSURE_AUTO_NOT_OFF", "GAIN_AUTO_NOT_OFF",
    "ORIGINAL_EXPOSURE_UNEXPECTED", "PLANNED_EXPOSURE_OUT_OF_RANGE", "EXPOSURE_NOT_WRITEABLE",
) + exposure_info.VALUE_STATUSES
NOT_OFF_STATUS = {
    "TriggerMode": "TRIGGER_MODE_NOT_OFF",
    "ExposureAuto": "EXPOSURE_AUTO_NOT_OFF",
    "GainAuto": "GAIN_AUTO_NOT_OFF",
}

RESTORE_NOT_REQUIRED = "NOT_REQUIRED"
RESTORE_VERIFIED = "RESTORED_AND_VERIFIED"
RESTORE_WRITE_FAILED = "RESTORE_WRITE_FAILED"
RESTORE_READBACK_FAILED = "RESTORE_READBACK_FAILED"
RESTORE_MISMATCH = "RESTORE_READBACK_MISMATCH"

SCOPE_NOTE = (
    "Two uncalibrated frames. DN values are raw digital numbers; no photometric, radiometric, "
    "sensitivity or noise quantity is derived. Frame bytes are held in memory only and no "
    "image or log file was saved."
)


class SweepError(first_frame.CaptureError):
    """Controlled sweep failure, tagged with a status and optional native code."""


@dataclass(frozen=True)
class AppliedExposure:
    requested: float
    set_code: int = None
    readback: float = None


@dataclass(frozen=True)
class SweepFrame:
    index: int
    requested_exposure: float
    readback_exposure: float
    frame_id: int
    width: int
    height: int
    pixel_format: int
    pixel_format_label: str
    padding_x: int
    padding_y: int
    payload_bytes: int
    min_dn: int
    max_dn: int
    mean_dn: float
    saturation_fraction: float
    acquisition_source: str = SOURCE_HARDWARE


@dataclass(frozen=True)
class RestoreOutcome:
    status: str
    original: float = None
    set_code: int = None
    readback: float = None
    error: str = None

    @property
    def ok(self):
        return self.status in (RESTORE_NOT_REQUIRED, RESTORE_VERIFIED)


@dataclass
class SweepResult:
    status: str
    message: str = ""
    sdk_code: int = None
    sdk_code_name: str = None
    device: first_frame.SelectedDevice = None
    features: dict = field(default_factory=dict)
    original: exposure_info.ExposureRange = None
    applied: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    restore: RestoreOutcome = None
    steps: list = field(default_factory=list)

    @property
    def cleanup_ok(self):
        return all(step.ok for step in self.steps)

    @property
    def restore_ok(self):
        return self.restore is not None and self.restore.ok

    @property
    def ok(self):
        return (
            self.status == STATUS_COMPLETE
            and len(self.frames) == len(PLANNED_EXPOSURES)
            and self.restore is not None
            and self.restore.status == RESTORE_VERIFIED
            and self.cleanup_ok
        )


def exposure_matches(actual, expected):
    return math.isfinite(actual) and abs(actual - expected) <= EXPOSURE_TOLERANCE


def require_mono8_layout(sdk, meta):
    """Accept only genuine, unpadded Mono8 whose payload is exactly width*height. Reads no pixel memory."""
    if meta["pixel_format"] != sdk.IMV_EPixelType.gvspPixelMono8:
        raise SweepError("NOT_MONO8", f"pixel format 0x{meta['pixel_format'] & 0xFFFFFFFF:08X} is not Mono8")
    if meta["padding_x"] != 0 or meta["padding_y"] != 0:
        raise SweepError("INVALID_LAYOUT", "Mono8 frame reports padding; row layout not verified")
    if meta["size"] != meta["width"] * meta["height"]:
        raise SweepError("INVALID_LAYOUT", "Mono8 payload size does not equal width*height")


class _Session:
    """One camera session. Tracks acquired native resources so cleanup releases exactly those."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.camera = None
        self.handle_created = False
        self.opened = False
        self.grabbing = False
        self.frame = None
        self.frame_outstanding = False
        self.frame_index = None
        self.original_exposure = None
        self.write_issued = False
        self.interrupted = False
        self.steps = []

    def _check(self, status, name, code):
        if code != self.sdk.IMV_OK:
            raise SweepError(status, f"{name} failed", sdk_code=code)

    def _lifecycle_call(self, name, call):
        try:
            code = call()
        except KeyboardInterrupt:
            self.interrupted = True
            step = first_frame.CleanupStep(name, error="interrupted by KeyboardInterrupt")
        except Exception as exc:  # native wrapper failure must not skip the remaining cleanup
            step = first_frame.CleanupStep(name, error=repr(exc))
        else:
            step = first_frame.CleanupStep(name, code=code)
        self.steps.append(step)
        return step

    def read_off_feature(self, name):
        if not self.camera.IMV_FeatureIsAvailable(name):
            raise SweepError("FEATURE_UNAVAILABLE", f"{name} is not available; refusing to write")
        if not self.camera.IMV_FeatureIsReadable(name):
            raise SweepError("FEATURE_UNAVAILABLE", f"{name} is not readable; refusing to write")
        symbol = self.sdk.IMV_String()
        code = self.camera.IMV_GetEnumFeatureSymbol(name, symbol)
        self._check("FEATURE_UNAVAILABLE", f"IMV_GetEnumFeatureSymbol({name})", code)
        value = discovery.decode_field(symbol.str)
        if not value:
            raise SweepError("FEATURE_UNAVAILABLE", f"{name} returned an empty symbol; refusing to write")
        return value

    def read_exposure(self, getter_name, status="EXPOSURE_QUERY_FAILED"):
        value = ctypes.c_double(0.0)
        code = getattr(self.camera, getter_name)(EXPOSURE_FEATURE, value)
        self._check(status, f"{getter_name}({EXPOSURE_FEATURE})", code)
        return float(value.value)

    def verify_preconditions(self, result):
        for name in REQUIRED_OFF_FEATURES:
            value = self.read_off_feature(name)
            result.features[name] = value
            if value != "Off":
                raise SweepError(NOT_OFF_STATUS[name], f"{name} is {value!r}, required 'Off'; nothing was written")

        original = exposure_info.ExposureRange(
            current=self.read_exposure("IMV_GetDoubleFeatureValue"),
            minimum=self.read_exposure("IMV_GetDoubleFeatureMin"),
            maximum=self.read_exposure("IMV_GetDoubleFeatureMax"),
        )
        result.original = original
        exposure_info.validate_exposure(original)
        if not exposure_matches(original.current, EXPECTED_ORIGINAL_EXPOSURE):
            raise SweepError(
                "ORIGINAL_EXPOSURE_UNEXPECTED",
                f"original {EXPOSURE_FEATURE} {original.current!r} != {EXPECTED_ORIGINAL_EXPOSURE!r}; nothing was written",
            )
        for planned in PLANNED_EXPOSURES:
            if not original.minimum <= planned <= original.maximum:
                raise SweepError(
                    "PLANNED_EXPOSURE_OUT_OF_RANGE",
                    f"planned {planned!r} outside SDK range [{original.minimum!r}, {original.maximum!r}]",
                )
        if not self.camera.IMV_FeatureIsWriteable(EXPOSURE_FEATURE):
            raise SweepError("EXPOSURE_NOT_WRITEABLE", f"{EXPOSURE_FEATURE} is not writeable")
        self.original_exposure = original.current

    def apply_exposure(self, requested, result):
        self.write_issued = True  # set before the call: a failed or raising write may still have changed state
        code = self.camera.IMV_SetDoubleFeatureValue(EXPOSURE_FEATURE, requested)
        if code != self.sdk.IMV_OK:
            result.applied.append(AppliedExposure(requested, set_code=code))
            raise SweepError("EXPOSURE_WRITE_FAILED", f"IMV_SetDoubleFeatureValue({EXPOSURE_FEATURE}, {requested!r}) failed",
                             sdk_code=code)
        readback = self.read_exposure("IMV_GetDoubleFeatureValue", status="EXPOSURE_READBACK_FAILED")
        result.applied.append(AppliedExposure(requested, set_code=code, readback=readback))
        if not exposure_matches(readback, requested):
            raise SweepError("EXPOSURE_READBACK_MISMATCH",
                             f"requested {EXPOSURE_FEATURE} {requested!r}, SDK read back {readback!r}")
        return readback

    def release_frame(self):
        if not self.frame_outstanding:
            return None
        self.frame_outstanding = False
        frame, self.frame = self.frame, None
        return self._lifecycle_call(f"IMV_ReleaseFrame (frame {self.frame_index})",
                                    lambda: self.camera.IMV_ReleaseFrame(frame))

    def stop_grabbing(self):
        if not self.grabbing:
            return None
        self.grabbing = False
        return self._lifecycle_call(f"IMV_StopGrabbing (frame {self.frame_index})", self.camera.IMV_StopGrabbing)

    def acquire_frame(self, index, requested, readback):
        self.frame_index = index
        self._check("START_GRABBING_FAILED", "IMV_StartGrabbing", self.camera.IMV_StartGrabbing())
        self.grabbing = True

        frame = self.sdk.IMV_Frame()
        code = self.camera.IMV_GetFrame(frame, first_frame.GET_FRAME_TIMEOUT_MS)
        self._check("GET_FRAME_FAILED", f"IMV_GetFrame (frame {index}, timeout {first_frame.GET_FRAME_TIMEOUT_MS} ms)",
                    code)
        self.frame = frame
        self.frame_outstanding = True

        meta = first_frame.validate_frame_metadata(self.sdk, frame)
        require_mono8_layout(self.sdk, meta)
        data = ctypes.string_at(frame.pData, meta["size"])  # Python-owned copy before release
        if len(data) != meta["size"]:
            raise SweepError("INVALID_LAYOUT", "copied byte count does not match the frame size")

        for step in (self.release_frame(), self.stop_grabbing()):
            if not step.ok:
                raise SweepError("FRAME_LIFECYCLE_FAILED", f"{step.name} did not succeed")

        pixels = np.frombuffer(data, dtype=np.uint8).reshape(meta["height"], meta["width"])
        stats = analyze_frame(pixels, MONO8_BIT_DEPTH)
        return SweepFrame(
            index=index,
            requested_exposure=requested,
            readback_exposure=readback,
            frame_id=meta["frame_id"],
            width=meta["width"],
            height=meta["height"],
            pixel_format=meta["pixel_format"],
            pixel_format_label=meta["label"] or "unrecognized",
            padding_x=meta["padding_x"],
            padding_y=meta["padding_y"],
            payload_bytes=len(data),
            min_dn=stats["min_dn"],
            max_dn=stats["max_dn"],
            mean_dn=stats["mean_dn"],
            saturation_fraction=stats["saturation_fraction"],
        )

    def run(self, result):
        device = first_frame.select_device(self.sdk)
        result.device = device

        self.camera = self.sdk.MvCamera()
        camera_key = ctypes.c_char_p(device.camera_key)
        code = self.camera.IMV_CreateHandle(self.sdk.IMV_ECreateHandleMode.modeByCameraKey, camera_key)
        self._check("CREATE_HANDLE_FAILED", "IMV_CreateHandle", code)
        self.handle_created = True

        self._check("OPEN_FAILED", "IMV_Open", self.camera.IMV_Open())
        self.opened = True

        self.verify_preconditions(result)
        for index, requested in enumerate(PLANNED_EXPOSURES, start=1):
            readback = self.apply_exposure(requested, result)
            result.frames.append(self.acquire_frame(index, requested, readback))
        result.status = STATUS_COMPLETE

    def restore_exposure(self):
        if not self.write_issued:
            return RestoreOutcome(RESTORE_NOT_REQUIRED, original=self.original_exposure)
        original = self.original_exposure
        try:
            code = self.camera.IMV_SetDoubleFeatureValue(EXPOSURE_FEATURE, original)
        except KeyboardInterrupt:
            self.interrupted = True
            return RestoreOutcome(RESTORE_WRITE_FAILED, original=original, error="interrupted by KeyboardInterrupt")
        except Exception as exc:
            return RestoreOutcome(RESTORE_WRITE_FAILED, original=original, error=repr(exc))
        if code != self.sdk.IMV_OK:
            return RestoreOutcome(RESTORE_WRITE_FAILED, original=original, set_code=code)

        value = ctypes.c_double(0.0)
        try:
            read_code = self.camera.IMV_GetDoubleFeatureValue(EXPOSURE_FEATURE, value)
        except KeyboardInterrupt:
            self.interrupted = True
            return RestoreOutcome(RESTORE_READBACK_FAILED, original=original, set_code=code,
                                  error="interrupted by KeyboardInterrupt")
        except Exception as exc:
            return RestoreOutcome(RESTORE_READBACK_FAILED, original=original, set_code=code, error=repr(exc))
        if read_code != self.sdk.IMV_OK:
            return RestoreOutcome(RESTORE_READBACK_FAILED, original=original, set_code=code,
                                  error=f"IMV_GetDoubleFeatureValue returned {read_code}")
        readback = float(value.value)
        status = RESTORE_VERIFIED if exposure_matches(readback, original) else RESTORE_MISMATCH
        return RestoreOutcome(status, original=original, set_code=code, readback=readback)

    def cleanup(self):
        """Release frame, stop grabbing, restore exposure, close, destroy. Every applicable step is attempted."""
        self.release_frame()
        self.stop_grabbing()
        restore = RestoreOutcome(RESTORE_NOT_REQUIRED)
        if self.opened:
            restore = self.restore_exposure()
            self.opened = False
            self._lifecycle_call("IMV_Close", self.camera.IMV_Close)
        if self.handle_created:
            self.handle_created = False
            self._lifecycle_call("IMV_DestroyHandle", self.camera.IMV_DestroyHandle)
        return restore, list(self.steps)


def run_exposure_sweep(sdk):
    """Run the one-session two-frame sweep. Never raises for SDK failures.

    Only a KeyboardInterrupt that escapes cleanup itself can propagate.
    """
    session = _Session(sdk)
    result = SweepResult(status=STATUS_UNEXPECTED_ERROR)
    try:
        session.run(result)
    except first_frame.CaptureError as exc:
        result.status = exc.status
        result.message = str(exc)
        result.sdk_code = exc.sdk_code
        if exc.sdk_code is not None:
            result.sdk_code_name = first_frame.error_label(sdk, exc.sdk_code)
    except KeyboardInterrupt:
        result.status = STATUS_INTERRUPTED
        result.message = "interrupted before the sweep completed"
    except Exception as exc:
        result.status = STATUS_UNEXPECTED_ERROR
        result.message = repr(exc)
    finally:
        result.restore, result.steps = session.cleanup()
    if session.interrupted and result.status != STATUS_INTERRUPTED:
        result.message = f"{result.status}: {result.message}" if result.message else "interrupted during cleanup"
        result.status = STATUS_INTERRUPTED
    return result


def _format_step(step):
    if step.ok:
        return f"{step.name}: OK"
    if step.error is not None:
        return f"{step.name}: FAILED ({step.error})"
    return f"{step.name}: FAILED with code {step.code}"


def format_plan():
    first, second = PLANNED_EXPOSURES
    return [
        f"Device required:     exactly one USB3 {EXPECTED_MODEL}",
        "Preconditions:       TriggerMode, ExposureAuto and GainAuto must read 'Off'",
        f"                     original {EXPOSURE_FEATURE} must read {EXPECTED_ORIGINAL_EXPOSURE!r} "
        "within the SDK min/max",
        f"Frame 1:             set {EXPOSURE_FEATURE} = {first!r} us, read back, acquire ONE frame",
        f"Frame 2:             set {EXPOSURE_FEATURE} = {second!r} us, read back, acquire ONE frame",
        f"Restore:             write back the originally read {EXPOSURE_FEATURE} and verify by readback,",
        "                     also after any failure, then Close and DestroyHandle",
        f"Unit basis:          {UNIT_BASIS}",
        "No other feature is written. No retries. No files are saved.",
    ]


def format_result(result):
    lines = []
    if result.device is not None:
        lines.append(f"Model:               {result.device.model}")
        lines.append(f"Serial:              {result.device.masked_serial}")
    for name, value in result.features.items():
        lines.append(f"{name + ' (read):':<21}{value}")
    if result.original is not None:
        original = result.original
        lines.append(f"Original {EXPOSURE_FEATURE}: {original.current!r} (SDK min {original.minimum!r}, "
                     f"max {original.maximum!r})")
        lines.append(f"Unit basis:          {UNIT_BASIS}")
    for applied in result.applied:
        readback = "not read" if applied.readback is None else repr(applied.readback)
        set_outcome = "IMV_OK" if applied.set_code == 0 else f"FAILED with code {applied.set_code}"
        lines.append(f"Set {EXPOSURE_FEATURE} {applied.requested!r}: {set_outcome}; read back {readback}")
    for frame in result.frames:
        lines.extend([
            f"Frame {frame.index}: requested {frame.requested_exposure!r}, read back {frame.readback_exposure!r}",
            f"  Frame ID (blockId): {frame.frame_id}",
            f"  Layout:             {frame.width} x {frame.height} "
            f"0x{frame.pixel_format & 0xFFFFFFFF:08X} ({frame.pixel_format_label}), "
            f"padding {frame.padding_x}/{frame.padding_y}, {frame.payload_bytes} bytes",
            f"  Source:             {frame.acquisition_source}",
            f"  min DN {frame.min_dn}, max DN {frame.max_dn}, mean DN {frame.mean_dn:.4f}, "
            f"saturation fraction {frame.saturation_fraction:.6f} (pixels == 255 DN)",
        ])
    if len(result.frames) == 2:
        first, second = result.frames
        lines.append(f"Observed mean DN change (frame 2 - frame 1): {second.mean_dn - first.mean_dn:+.4f} DN")
        if first.mean_dn > 0:
            lines.append(f"Observed mean DN ratio (frame 2 / frame 1): {second.mean_dn / first.mean_dn:.4f} "
                         f"(read-back exposure ratio {second.readback_exposure / first.readback_exposure:.4f}; "
                         "linearity not assumed)")
        else:
            lines.append("Observed mean DN ratio: undefined (frame 1 mean is 0 DN)")
    restore = result.restore
    if restore is not None:
        detail = restore.status
        if restore.status != RESTORE_NOT_REQUIRED:
            detail += f" (original {restore.original!r}"
            if restore.set_code not in (None, 0):
                detail += f", write code {restore.set_code}"
            if restore.readback is not None:
                detail += f", read back {restore.readback!r}"
            if restore.error is not None:
                detail += f", {restore.error}"
            detail += ")"
        lines.append(f"Exposure restoration: {detail}")
    for step in result.steps:
        lines.append(f"Lifecycle {_format_step(step)}")

    if result.ok:
        lines.append("SWEEP_STATUS: SUCCESS")
        return lines
    detail = result.message
    if result.sdk_code is not None:
        detail += f" (error code {result.sdk_code} {result.sdk_code_name})"
    lines.append(f"SWEEP_FAILURE: {result.status}: {detail}")
    if not result.restore_ok:
        lines.append(f"RESTORE_FAILURE: {EXPOSURE_FEATURE} may not equal its original value; check in MV Viewer")
    if not result.cleanup_ok:
        lines.append("CLEANUP_FAILURE: one or more SDK release/shutdown calls failed (see above)")
    lines.append("SWEEP_STATUS: FAILED")
    return lines


def exit_code_for(result):
    if result.ok:
        return EXIT_OK
    if result.status == STATUS_INTERRUPTED:
        return EXIT_INTERRUPTED
    if not result.restore_ok:
        return EXIT_RESTORE_FAILED
    if not result.cleanup_ok:
        return EXIT_CLEANUP_FAILED
    if result.status in first_frame.DEVICE_SELECTION_STATUSES:
        return EXIT_DEVICE_SELECTION
    if result.status in PRECONDITION_STATUSES:
        return EXIT_PRECONDITION
    return EXIT_SWEEP_FAILED


def main(argv=None, sdk_loader=None, input_fn=input, stdin=None):
    parser = argparse.ArgumentParser(
        description="Two-frame ExposureTime sweep (1000 us, 5000 us) on the iRAYPLE MV-A7200MU130."
    )
    parser.add_argument("--sweep-once", action="store_true",
                        help="required: run the single two-frame session (interactive confirmation follows)")
    parser.add_argument("--unit-us-verified", action="store_true",
                        help="required: the operator confirmed in MV Viewer that ExposureTime is in microseconds")
    args = parser.parse_args(argv)

    print("LUMEN iRAYPLE exposure sweep (two frames, one session)")
    if not (args.sweep_once and args.unit_us_verified):
        print("NO_ACTION: both --sweep-once and --unit-us-verified are required. No camera was accessed.")
        return EXIT_REFUSED

    stdin = sys.stdin if stdin is None else stdin
    if not first_frame._is_interactive(stdin):
        print("REFUSED: an interactive terminal is required for confirmation. No camera was accessed.")
        return EXIT_REFUSED
    for line in format_plan():
        print(line)
    try:
        response = input_fn(f"Type {CONFIRMATION_PHRASE} to run this experiment: ")
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
        print(f"SWEEP_FAILURE: SDK_LOAD_FAILED: {exc}")
        print("SWEEP_STATUS: FAILED")
        return EXIT_SWEEP_FAILED

    try:
        result = run_exposure_sweep(sdk)
    except KeyboardInterrupt:
        print("SWEEP_FAILURE: INTERRUPTED during cleanup; camera state and ExposureTime unknown.")
        print("SWEEP_STATUS: FAILED")
        return EXIT_INTERRUPTED
    for line in format_result(result):
        print(line)
    if result.frames:
        print(SCOPE_NOTE)
    return exit_code_for(result)


if __name__ == "__main__":
    sys.exit(main())
