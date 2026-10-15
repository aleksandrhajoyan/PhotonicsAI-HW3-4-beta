"""IRaypleCamera: CameraBackend adapter for the iRAYPLE MV-A7200MU130 over the official HuaRay IMV SDK.

Loading:
    - Importing this module loads nothing from the vendor SDK.
    - The already loaded vendor binding (the IMVApi module) is injected through
      the constructor. Constructing the backend makes no SDK call.
    - No backend method imports a vendor module, loads a DLL or changes the
      process working directory. preload_vendor_sdk() is the only route to the
      existing loader (irayple_discovery.load_vendor_sdk, which briefly changes
      the working directory). A trusted host calls it once at startup, before
      any MCP tool is served; it must never be called from a tool handler.

Lifecycle (one camera per instance, sequential SDK use only):
    open()             one USB3 enumeration; exactly one MV-A7200MU130; CreateHandle
                       by camera key; Open; TriggerMode, ExposureAuto and GainAuto
                       must read Off; ExposureTime current/min/max read and
                       validated. No feature write, no grabbing.
    set_exposure_us()  re-check the Off features and writeability, mark restoration
                       required, IMV_SetDoubleFeatureValue(ExposureTime), read back.
                       The readback must equal the request exactly; nothing is
                       clamped, rounded or retried.
    capture_frame()    StartGrabbing, one IMV_GetFrame (bounded timeout, never
                       retried), validate unpadded Mono8, copy into NumPy-owned
                       memory, ReleaseFrame once, StopGrabbing. A CameraFrame is
                       returned only when every step succeeded.
    close()            release/stop anything still outstanding, restore and verify
                       the original ExposureTime if any write was attempted, Close,
                       DestroyHandle. Failures raise IRaypleCloseError.

Frame identifiers: CameraFrame.frame_id is this instance's successful-acquisition
counter (1, 2, 3, ...). Grabbing restarts and reopening never reset it, and it is
not a camera-native identifier. The SDK's IMV_FrameInfo.blockId (the real camera
reported 1 on successive single-frame acquisitions) is kept separately as
get_info()["last_native_block_id"].

ExposureTime unit: microseconds, per the operator's MV Viewer check recorded for
HW3 (irayple_exposure_sweep --unit-us-verified); the installed SDK headers do not
state it.

Concurrency: every SDK-touching method holds a non-blocking lock; an overlapping or
re-entrant call is rejected with RuntimeError, never queued. No thread-safety or
thread-affinity property of the vendor SDK is assumed.
"""

import contextlib
import ctypes
import math
import threading
from numbers import Real

import numpy as np

import irayple_discovery as discovery
import irayple_exposure_info as exposure_info
import irayple_exposure_sweep as exposure_sweep
import irayple_first_frame as first_frame
from camera_contract import SOURCE_HARDWARE, CameraFrame

EXPECTED_MODEL = discovery.EXPECTED_MODEL
EXPOSURE_FEATURE = exposure_info.EXPOSURE_FEATURE
REQUIRED_OFF_FEATURES = exposure_sweep.REQUIRED_OFF_FEATURES
NOT_OFF_STATUS = exposure_sweep.NOT_OFF_STATUS
GET_FRAME_TIMEOUT_MS = first_frame.GET_FRAME_TIMEOUT_MS
BIT_DEPTH = 8
PIXEL_FORMAT = "Mono8"
EXPOSURE_UNIT_BASIS = exposure_sweep.UNIT_BASIS
FRAME_ID_BASIS = "per-backend successful-acquisition counter; not a camera-native identifier"
REQUIRED_SDK_ATTRIBUTES = (
    "MvCamera", "IMV_DeviceList", "IMV_Frame", "IMV_String", "IMV_OK",
    "IMV_EInterfaceType", "IMV_ECreateHandleMode", "IMV_EPixelType",
)

RestoreOutcome = exposure_sweep.RestoreOutcome
RESTORE_NOT_REQUIRED = exposure_sweep.RESTORE_NOT_REQUIRED
RESTORE_VERIFIED = exposure_sweep.RESTORE_VERIFIED
RESTORE_WRITE_FAILED = exposure_sweep.RESTORE_WRITE_FAILED
RESTORE_READBACK_FAILED = exposure_sweep.RESTORE_READBACK_FAILED
RESTORE_MISMATCH = exposure_sweep.RESTORE_MISMATCH

STATUS_RESTORE_FAILED = "RESTORE_FAILED"
STATUS_CLEANUP_FAILED = "CLEANUP_FAILED"
STATUS_RESTORE_AND_CLEANUP_FAILED = "RESTORE_AND_CLEANUP_FAILED"


def preload_vendor_sdk():
    """Trusted-host startup only: load the vendor IMVApi binding through the existing loader.

    Call once, before any MCP tool is served, and pass the result to IRaypleCamera.
    The existing loader briefly changes the process working directory, so this must
    never run inside a backend operation or an MCP tool handler. Raises
    irayple_discovery.DiscoveryError when the SDK cannot be loaded.
    """
    return discovery.load_vendor_sdk()


class IRaypleCameraError(RuntimeError):
    """Controlled backend failure with a status and the native SDK return code, if any.

    cleanup_error is set when a failed open() could not release what it acquired.
    """

    def __init__(self, status, message, sdk_code=None, sdk_code_name=None):
        detail = message if sdk_code is None else f"{message} (error code {sdk_code} {sdk_code_name})"
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.sdk_code = sdk_code
        self.sdk_code_name = sdk_code_name
        self.cleanup_error = None


class IRaypleCloseError(IRaypleCameraError):
    """Shutdown did not fully succeed. restore and steps say which part failed."""

    def __init__(self, restore, steps, repeated=False):
        self.restore = restore
        self.steps = tuple(steps)
        parts = []
        if not self.restore_ok:
            detail = f"exposure restoration {restore.status} (original {restore.original!r}"
            if restore.set_code not in (None, 0):
                detail += f", write code {restore.set_code}"
            if restore.readback is not None:
                detail += f", read back {restore.readback!r}"
            if restore.error is not None:
                detail += f", {restore.error}"
            parts.append(detail + ")")
        failed = [_format_step(step) for step in self.steps if not step.ok]
        if failed:
            parts.append("cleanup failed: " + ", ".join(failed))
        if repeated:
            parts.append("reported by an earlier close(); native calls are not repeated")
        if not self.restore_ok and failed:
            status = STATUS_RESTORE_AND_CLEANUP_FAILED
        elif not self.restore_ok:
            status = STATUS_RESTORE_FAILED
        else:
            status = STATUS_CLEANUP_FAILED
        super().__init__(status, "; ".join(parts))

    @property
    def restore_ok(self):
        return self.restore.ok

    @property
    def cleanup_ok(self):
        return all(step.ok for step in self.steps)


def _format_step(step):
    if step.error is not None:
        return f"{step.name} FAILED ({step.error})"
    return f"{step.name} FAILED with code {step.code}"


class IRaypleCamera:
    """Real-hardware CameraBackend. Every returned frame is labelled source="hardware"."""

    def __init__(self, sdk):
        if sdk is None:
            raise TypeError("sdk must be the already loaded HuaRay IMVApi binding, got None")
        missing = [name for name in REQUIRED_SDK_ATTRIBUTES if not hasattr(sdk, name)]
        if missing:
            raise TypeError(f"sdk is not a HuaRay IMVApi binding; missing {missing}")
        # No SDK call here: nothing touches the camera before open().
        self._sdk = sdk
        self._lock = threading.Lock()
        self._camera = None
        self._device = None
        self._handle_created = False
        self._opened = False
        self._is_open = False
        self._grabbing = False
        self._frame = None
        self._limits = None
        self._original_exposure = None
        self._exposure_us = None  # last SDK-verified ExposureTime; None until verified
        self._restore_required = False
        self._unresolved_close = None
        self._frames_acquired = 0
        self._last_native_block_id = None

    @property
    def is_open(self):
        return self._is_open

    # -- CameraBackend -------------------------------------------------------

    def open(self):
        with self._exclusive("open"):
            if self._is_open:
                return
            if self._unresolved_close is not None:
                raise IRaypleCameraError(
                    "UNRESOLVED_CLEANUP_FAILURE",
                    f"an earlier shutdown did not complete ({self._unresolved_close}); refusing to reopen",
                )
            try:
                try:
                    self._open_and_verify()
                except first_frame.CaptureError as exc:
                    raise self._error(exc.status, str(exc), exc.sdk_code) from exc
            except BaseException as exc:
                restore, steps = self._shutdown(restore=False)
                if not all(step.ok for step in steps):
                    self._unresolved_close = IRaypleCloseError(restore, steps)
                    if isinstance(exc, IRaypleCameraError):
                        exc.cleanup_error = self._unresolved_close
                    exc.add_note(f"partial-open cleanup also failed: {self._unresolved_close}")
                raise

    def close(self):
        with self._exclusive("close"):
            if self._unresolved_close is not None:
                previous = self._unresolved_close
                raise IRaypleCloseError(previous.restore, previous.steps, repeated=True)
            if not (self._handle_created or self._opened or self._grabbing or self._frame is not None):
                self._is_open = False
                return
            restore, steps = self._shutdown(restore=True)
            if not restore.ok or not all(step.ok for step in steps):
                self._unresolved_close = IRaypleCloseError(restore, steps)
                raise self._unresolved_close

    def get_info(self):
        """Describe the backend from cached state. Makes no SDK call."""
        device = self._device
        return {
            "backend": "IRaypleCamera",
            "source": SOURCE_HARDWARE,
            "synthetic": False,
            "description": "iRAYPLE camera via the official HuaRay IMV SDK. Uncalibrated raw DN.",
            "model": EXPECTED_MODEL if device is None else device.model,
            "serial": None if device is None else device.masked_serial,
            "bit_depth": BIT_DEPTH,
            "pixel_format": PIXEL_FORMAT,
            "exposure_limits_us": None if self._limits is None else list(self._limits),
            "exposure_us": self._exposure_us,
            "exposure_unit_basis": EXPOSURE_UNIT_BASIS,
            "exposure_restore_required": self._restore_required,
            "is_open": self._is_open,
            "frames_acquired": self._frames_acquired,
            "frame_id_basis": FRAME_ID_BASIS,
            "last_native_block_id": self._last_native_block_id,
            "unresolved_cleanup_failure": None if self._unresolved_close is None else str(self._unresolved_close),
        }

    def get_exposure_limits_us(self):
        """Return the SDK-reported (min, max) read at open(). Makes no SDK call."""
        self._require_open("get_exposure_limits_us")
        return self._limits

    def set_exposure_us(self, value):
        with self._exclusive("set_exposure_us"):
            self._require_open("set_exposure_us")
            value = self._validate_request(value)
            self._require_idle("set_exposure_us")
            for name in REQUIRED_OFF_FEATURES:
                self._require_off(name)
            if not self._camera.IMV_FeatureIsWriteable(EXPOSURE_FEATURE):
                raise self._error("EXPOSURE_NOT_WRITEABLE", f"{EXPOSURE_FEATURE} is not writeable")
            # Set before the call: a failed or raising write may still have changed the camera.
            self._restore_required = True
            self._exposure_us = None
            code = self._camera.IMV_SetDoubleFeatureValue(EXPOSURE_FEATURE, value)
            self._check("EXPOSURE_WRITE_FAILED", f"IMV_SetDoubleFeatureValue({EXPOSURE_FEATURE}, {value!r})", code)
            readback = self._read_exposure("IMV_GetDoubleFeatureValue", "EXPOSURE_READBACK_FAILED")
            if readback != value:
                raise self._error(
                    "EXPOSURE_READBACK_MISMATCH",
                    f"requested {EXPOSURE_FEATURE} {value!r} us, SDK read back {readback!r} us; "
                    "not accepted and not retried",
                )
            self._exposure_us = readback
            return readback

    def capture_frame(self):
        with self._exclusive("capture_frame"):
            self._require_open("capture_frame")
            if self._exposure_us is None:
                raise self._error(
                    "EXPOSURE_UNVERIFIED",
                    f"{EXPOSURE_FEATURE} has no verified readback since the last write attempt; refusing to acquire",
                )
            self._require_idle("capture_frame")
            exposure_us = self._exposure_us
            try:
                pixels, native_block_id = self._acquire()
            except BaseException as exc:
                failed = [step for step in self._end_acquisition() if not step.ok]
                if failed:
                    exc.add_note("acquisition cleanup also failed: " + ", ".join(map(_format_step, failed)))
                raise
            failed = [step for step in self._end_acquisition() if not step.ok]
            if failed:
                raise self._error(
                    "FRAME_LIFECYCLE_FAILED",
                    "frame discarded because " + ", ".join(map(_format_step, failed)),
                )
            self._frames_acquired += 1
            self._last_native_block_id = native_block_id
            return CameraFrame(
                pixels=pixels,
                bit_depth=BIT_DEPTH,
                exposure_us=exposure_us,
                source=SOURCE_HARDWARE,
                frame_id=self._frames_acquired,
            )

    # -- open ----------------------------------------------------------------

    def _open_and_verify(self):
        device = first_frame.select_device(self._sdk)
        self._camera = self._sdk.MvCamera()
        code = self._camera.IMV_CreateHandle(
            self._sdk.IMV_ECreateHandleMode.modeByCameraKey, ctypes.c_char_p(device.camera_key)
        )
        self._check("CREATE_HANDLE_FAILED", "IMV_CreateHandle", code)
        self._handle_created = True

        self._check("OPEN_FAILED", "IMV_Open", self._camera.IMV_Open())
        self._opened = True

        for name in REQUIRED_OFF_FEATURES:
            self._require_off(name)
        exposure = exposure_info.ExposureRange(
            current=self._read_exposure("IMV_GetDoubleFeatureValue"),
            minimum=self._read_exposure("IMV_GetDoubleFeatureMin"),
            maximum=self._read_exposure("IMV_GetDoubleFeatureMax"),
        )
        exposure_info.validate_exposure(exposure)

        self._device = device
        self._limits = (exposure.minimum, exposure.maximum)
        self._original_exposure = exposure.current
        self._exposure_us = exposure.current
        self._is_open = True

    def _require_off(self, name):
        if not self._camera.IMV_FeatureIsAvailable(name):
            raise self._error("FEATURE_UNAVAILABLE", f"{name} is not available")
        if not self._camera.IMV_FeatureIsReadable(name):
            raise self._error("FEATURE_UNAVAILABLE", f"{name} is not readable")
        symbol = self._sdk.IMV_String()
        self._check("FEATURE_UNAVAILABLE", f"IMV_GetEnumFeatureSymbol({name})",
                    self._camera.IMV_GetEnumFeatureSymbol(name, symbol))
        value = discovery.decode_field(symbol.str)
        if value != "Off":
            raise self._error(NOT_OFF_STATUS[name], f"{name} is {value!r}, required 'Off'; left unchanged")

    def _read_exposure(self, getter_name, status="EXPOSURE_QUERY_FAILED"):
        value = ctypes.c_double(0.0)
        code = getattr(self._camera, getter_name)(EXPOSURE_FEATURE, value)
        self._check(status, f"{getter_name}({EXPOSURE_FEATURE})", code)
        return float(value.value)

    # -- exposure ------------------------------------------------------------

    def _validate_request(self, value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"exposure_us must be a real number, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"exposure_us must be finite, got {value}")
        minimum, maximum = self._limits
        if not minimum <= value <= maximum:
            raise ValueError(f"exposure_us {value} outside the SDK-reported range [{minimum}, {maximum}] us")
        return value

    # -- acquisition ---------------------------------------------------------

    def _acquire(self):
        """StartGrabbing and one GetFrame; copies the frame into owned memory. Does not release."""
        self._check("START_GRABBING_FAILED", "IMV_StartGrabbing", self._camera.IMV_StartGrabbing())
        self._grabbing = True

        frame = self._sdk.IMV_Frame()
        code = self._camera.IMV_GetFrame(frame, GET_FRAME_TIMEOUT_MS)
        self._check("GET_FRAME_FAILED", f"IMV_GetFrame (timeout {GET_FRAME_TIMEOUT_MS} ms, not retried)", code)
        self._frame = frame

        try:
            meta = first_frame.validate_frame_metadata(self._sdk, frame)
        except first_frame.CaptureError as exc:
            raise self._error(exc.status, str(exc), exc.sdk_code) from exc
        if meta["pixel_format"] != self._sdk.IMV_EPixelType.gvspPixelMono8:
            raise self._error("NOT_MONO8", f"pixel format 0x{meta['pixel_format'] & 0xFFFFFFFF:08X} is not Mono8")
        if meta["padding_x"] != 0 or meta["padding_y"] != 0:
            raise self._error("INVALID_LAYOUT", "Mono8 frame reports padding; row layout not verified")
        if meta["size"] != meta["width"] * meta["height"]:
            raise self._error("INVALID_LAYOUT", "Mono8 payload size does not equal width*height")

        # Copy into NumPy-owned memory before the SDK buffer is released.
        pixels = np.empty((meta["height"], meta["width"]), dtype=np.uint8)
        ctypes.memmove(pixels.ctypes.data, frame.pData, meta["size"])
        return pixels, meta["frame_id"]  # validate_frame_metadata's frame_id is the native blockId

    def _end_acquisition(self):
        """Release the outstanding frame exactly once, then stop grabbing if it is running."""
        steps = []
        if self._frame is not None:
            frame, self._frame = self._frame, None
            steps.append(self._native_step("IMV_ReleaseFrame", lambda: self._camera.IMV_ReleaseFrame(frame)))
        if self._grabbing:
            step = self._native_step("IMV_StopGrabbing", self._camera.IMV_StopGrabbing)
            if step.ok:
                self._grabbing = False
            steps.append(step)
        return steps

    # -- shutdown ------------------------------------------------------------

    def _shutdown(self, restore):
        """Release, stop, restore if required, Close, DestroyHandle. Every applicable step is attempted."""
        steps = self._end_acquisition()
        outcome = RestoreOutcome(RESTORE_NOT_REQUIRED, original=self._original_exposure)
        if self._opened and restore and self._restore_required:
            outcome = self._restore_exposure()
        if self._opened:
            self._opened = False
            steps.append(self._native_step("IMV_Close", self._camera.IMV_Close))
        if self._handle_created:
            self._handle_created = False
            steps.append(self._native_step("IMV_DestroyHandle", self._camera.IMV_DestroyHandle))
        self._camera = None
        self._device = None
        self._is_open = False
        self._grabbing = False
        self._limits = None
        self._original_exposure = None
        self._exposure_us = None
        self._restore_required = False
        return outcome, steps

    def _restore_exposure(self):
        original = self._original_exposure
        try:
            code = self._camera.IMV_SetDoubleFeatureValue(EXPOSURE_FEATURE, original)
        except Exception as exc:
            return RestoreOutcome(RESTORE_WRITE_FAILED, original=original, error=repr(exc))
        if code != self._sdk.IMV_OK:
            return RestoreOutcome(RESTORE_WRITE_FAILED, original=original, set_code=code)
        value = ctypes.c_double(0.0)
        try:
            read_code = self._camera.IMV_GetDoubleFeatureValue(EXPOSURE_FEATURE, value)
        except Exception as exc:
            return RestoreOutcome(RESTORE_READBACK_FAILED, original=original, set_code=code, error=repr(exc))
        if read_code != self._sdk.IMV_OK:
            return RestoreOutcome(RESTORE_READBACK_FAILED, original=original, set_code=code,
                                  error=f"IMV_GetDoubleFeatureValue returned {read_code}")
        readback = float(value.value)
        status = RESTORE_VERIFIED if readback == original else RESTORE_MISMATCH
        return RestoreOutcome(status, original=original, set_code=code, readback=readback)

    # -- helpers -------------------------------------------------------------

    @contextlib.contextmanager
    def _exclusive(self, operation):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                f"{operation} rejected: another IRaypleCamera operation is in progress (sequential use only)"
            )
        try:
            yield
        finally:
            self._lock.release()

    def _require_open(self, operation):
        if not self._is_open:
            raise RuntimeError(f"{operation} requires an open camera; call open() first")

    def _require_idle(self, operation):
        if self._grabbing or self._frame is not None:
            raise self._error("ACQUISITION_ACTIVE",
                              f"{operation} refused: grabbing did not stop cleanly; close() is required")

    def _check(self, status, name, code):
        if code != self._sdk.IMV_OK:
            raise self._error(status, f"{name} failed", code)

    def _error(self, status, message, code=None):
        name = None if code is None else first_frame.error_label(self._sdk, code)
        return IRaypleCameraError(status, message, sdk_code=code, sdk_code_name=name)

    def _native_step(self, name, call):
        try:
            code = call()
        except Exception as exc:  # a native wrapper failure must not skip the remaining cleanup
            return first_frame.CleanupStep(name, error=repr(exc))
        return first_frame.CleanupStep(name, code=code)
