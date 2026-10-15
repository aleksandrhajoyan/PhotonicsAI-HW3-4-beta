"""Manual one-frame smoke test of the IRaypleCamera CameraBackend against the physical iRAYPLE camera.

Nothing is loaded from the vendor SDK on import. main() touches the camera only with
--check-once, an interactive terminal and the typed confirmation phrase. Then:
irayple_camera.preload_vendor_sdk() once, one IRaypleCamera instance, and through its
public CameraBackend methods only: open, get_info, get_exposure_limits_us, verify the
device and the original ExposureTime (5000 us), set_exposure_us(1000.0), exactly one
capture_frame(), validate the CameraFrame, analyze it with measurements.analyze_frame,
and close() in a mandatory finally block. Exposure restoration is the backend's job:
close() restores and verifies the original ExposureTime whenever a write was issued and
raises IRaypleCloseError otherwise. Nothing is retried and nothing is saved.

If the process is killed (task manager, power loss, console closed) the finally block
does not run and restoration cannot be guaranteed; check ExposureTime in MV Viewer.
"""

import argparse
import math
import sys

import numpy as np

import irayple_camera as ic
import irayple_discovery as discovery
import irayple_exposure_sweep as exposure_sweep
import irayple_first_frame as first_frame
from camera_contract import SOURCE_HARDWARE, CameraFrame
from measurements import analyze_frame

CONFIRMATION_PHRASE = "TEST IRAYPLE BACKEND"
EXPECTED_MODEL = ic.EXPECTED_MODEL
EXPECTED_BACKEND = "IRaypleCamera"
EXPECTED_ORIGINAL_EXPOSURE_US = 5000.0
TARGET_EXPOSURE_US = 1000.0
# Reuses the sweep's justification: 1000.0 and 5000.0 are exact in binary32 and
# binary64, so the tolerance only absorbs representation round-off. The backend
# itself accepts a write only when the readback equals the request exactly.
EXPOSURE_TOLERANCE_US = exposure_sweep.EXPOSURE_TOLERANCE
EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1200
EXPECTED_BIT_DEPTH = ic.BIT_DEPTH
EXPECTED_FRAME_ID = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_CLEANUP_FAILED = 4
EXIT_PRECONDITION = 5
EXIT_RESTORE_FAILED = 6
EXIT_INTERRUPTED = 130

STATUS_COMPLETE = "SMOKE_TEST_COMPLETE"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
PRECONDITION_STATUSES = (
    "UNEXPECTED_BACKEND_INFO", "INVALID_EXPOSURE_LIMITS", "ORIGINAL_EXPOSURE_UNEXPECTED",
    "TARGET_EXPOSURE_OUT_OF_RANGE",
)

SCOPE_NOTE = (
    "One uncalibrated frame. DN values are raw digital numbers; no photometric or radiometric "
    "quantity is derived. Pixels were held in memory only; no image, raw data or report was saved."
)


class SmokeError(Exception):
    """Controlled smoke-test refusal or validation failure."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class SmokeResult:
    def __init__(self):
        self.status = STATUS_UNEXPECTED_ERROR
        self.message = ""
        self.model = None
        self.serial = None
        self.original_exposure_us = None
        self.limits_us = None
        self.requested_exposure_us = None
        self.applied_exposure_us = None
        self.capture_calls = 0
        self.width = None
        self.height = None
        self.dtype = None
        self.source = None
        self.frame_id = None
        self.native_block_id = None
        self.stats = None
        self.restore_required = None  # backend's exposure_restore_required just before close()
        self.close_attempted = False
        self.close_error = None
        self.interrupted = False

    @property
    def close_ok(self):
        return self.close_attempted and self.close_error is None

    @property
    def restore_ok(self):
        if not self.close_attempted:
            return False
        if self.close_error is None:
            return True
        return isinstance(self.close_error, ic.IRaypleCloseError) and self.close_error.restore_ok

    @property
    def ok(self):
        return (
            self.status == STATUS_COMPLETE
            and not self.interrupted
            and self.stats is not None
            and self.capture_calls == 1
            and self.close_ok
        )


def _validate_limits(limits):
    if not (isinstance(limits, tuple) and len(limits) == 2):
        raise SmokeError("INVALID_EXPOSURE_LIMITS", f"backend returned limits {limits!r}, expected (min, max)")
    minimum, maximum = limits
    for value in limits:
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
            raise SmokeError("INVALID_EXPOSURE_LIMITS", f"backend returned non-finite or non-float limits {limits!r}")
    if not 0 < minimum < maximum:
        raise SmokeError("INVALID_EXPOSURE_LIMITS", f"backend returned an invalid range {limits!r}")


def _verify_device(info):
    expected = {"backend": EXPECTED_BACKEND, "source": SOURCE_HARDWARE, "synthetic": False,
                "model": EXPECTED_MODEL, "is_open": True}
    for key, value in expected.items():
        if info.get(key) != value or type(info.get(key)) is not type(value):
            raise SmokeError("UNEXPECTED_BACKEND_INFO", f"get_info()[{key!r}] is {info.get(key)!r}, expected {value!r}")


def validate_frame(frame, applied_exposure_us):
    """Accept only a genuine hardware Mono8 CameraFrame of the expected native layout."""
    if not isinstance(frame, CameraFrame):
        raise SmokeError("INVALID_FRAME", f"capture_frame() returned {type(frame).__name__}, not CameraFrame")
    if frame.source != SOURCE_HARDWARE:
        raise SmokeError("NOT_HARDWARE_FRAME", f"frame source is {frame.source!r}, expected {SOURCE_HARDWARE!r}")
    if isinstance(frame.bit_depth, bool) or frame.bit_depth != EXPECTED_BIT_DEPTH:
        raise SmokeError("INVALID_FRAME", f"bit_depth {frame.bit_depth!r}, expected {EXPECTED_BIT_DEPTH}")
    pixels = frame.pixels
    if not isinstance(pixels, np.ndarray):
        raise SmokeError("INVALID_FRAME", f"pixels is {type(pixels).__name__}, not numpy.ndarray")
    if pixels.dtype != np.uint8:
        raise SmokeError("INVALID_FRAME", f"pixel dtype {pixels.dtype}, expected uint8")
    if pixels.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise SmokeError("INVALID_FRAME", f"pixel shape {pixels.shape}, expected ({EXPECTED_HEIGHT}, {EXPECTED_WIDTH})")
    if frame.exposure_us != applied_exposure_us or frame.exposure_us != TARGET_EXPOSURE_US:
        raise SmokeError("INVALID_FRAME", f"frame exposure_us {frame.exposure_us!r}, expected {TARGET_EXPOSURE_US!r}")
    if isinstance(frame.frame_id, bool) or not isinstance(frame.frame_id, int) or frame.frame_id != EXPECTED_FRAME_ID:
        raise SmokeError("INVALID_FRAME", f"frame_id {frame.frame_id!r}, expected {EXPECTED_FRAME_ID}")


def _run(camera, result):
    camera.open()
    info = camera.get_info()
    limits = camera.get_exposure_limits_us()
    result.model = info.get("model")
    result.serial = info.get("serial")  # masked by the backend
    result.original_exposure_us = info.get("exposure_us")
    result.limits_us = limits

    _verify_device(info)
    _validate_limits(limits)
    original = result.original_exposure_us
    if (isinstance(original, bool) or not isinstance(original, float)
            or not exposure_sweep.exposure_matches(original, EXPECTED_ORIGINAL_EXPOSURE_US)):
        raise SmokeError("ORIGINAL_EXPOSURE_UNEXPECTED",
                         f"original ExposureTime {original!r} us != {EXPECTED_ORIGINAL_EXPOSURE_US!r} us "
                         f"(tolerance {EXPOSURE_TOLERANCE_US}); nothing was written")
    minimum, maximum = limits
    if not minimum <= TARGET_EXPOSURE_US <= maximum:
        raise SmokeError("TARGET_EXPOSURE_OUT_OF_RANGE",
                         f"target {TARGET_EXPOSURE_US!r} us outside SDK range [{minimum!r}, {maximum!r}]; "
                         "nothing was written")

    result.requested_exposure_us = TARGET_EXPOSURE_US
    applied = camera.set_exposure_us(TARGET_EXPOSURE_US)
    result.applied_exposure_us = applied
    if isinstance(applied, bool) or not isinstance(applied, float) or applied != TARGET_EXPOSURE_US:
        raise SmokeError("EXPOSURE_NOT_APPLIED", f"set_exposure_us returned {applied!r}, expected {TARGET_EXPOSURE_US!r}")

    result.capture_calls += 1
    frame = camera.capture_frame()
    after = camera.get_info()
    result.native_block_id = after.get("last_native_block_id")
    result.source = frame.source
    result.frame_id = frame.frame_id
    if isinstance(frame.pixels, np.ndarray):
        result.height, result.width = frame.pixels.shape[:2] if frame.pixels.ndim >= 2 else (None, None)
        result.dtype = str(frame.pixels.dtype)
    validate_frame(frame, applied)
    if after.get("frames_acquired") != EXPECTED_FRAME_ID:
        raise SmokeError("INVALID_FRAME", f"backend reports {after.get('frames_acquired')!r} acquisitions, expected 1")

    stats = analyze_frame(frame.pixels, frame.bit_depth)
    result.stats = {key: stats[key] for key in ("min_dn", "max_dn", "mean_dn", "saturation_fraction")}
    result.status = STATUS_COMPLETE


def run_smoke_test(camera):
    """Run the one-frame procedure on one backend instance. close() is always attempted.

    Only a second KeyboardInterrupt arriving while close() itself handles one can propagate.
    """
    result = SmokeResult()
    try:
        _run(camera, result)
    except SmokeError as exc:
        result.status, result.message = exc.status, str(exc)
    except ic.IRaypleCameraError as exc:
        result.status, result.message = exc.status, str(exc)
    except KeyboardInterrupt:
        result.interrupted = True
        result.status, result.message = STATUS_INTERRUPTED, "interrupted before the smoke test completed"
    except Exception as exc:
        result.status, result.message = STATUS_UNEXPECTED_ERROR, repr(exc)
    finally:
        try:
            result.restore_required = camera.get_info().get("exposure_restore_required")
        except Exception:  # get_info() makes no SDK call; a failure here must not skip close()
            result.restore_required = None
        result.close_attempted = True
        try:
            camera.close()
        except KeyboardInterrupt:
            result.interrupted = True
            result.close_error = "interrupted by KeyboardInterrupt during close(); camera state unknown"
        except Exception as exc:
            result.close_error = exc
    return result


def format_plan():
    return [
        f"Device required:     exactly one USB3 {EXPECTED_MODEL}, via the existing IRaypleCamera backend",
        "SDK:                 irayple_camera.preload_vendor_sdk() once, then one IRaypleCamera instance",
        "Open:                camera.open() (TriggerMode, ExposureAuto, GainAuto must read 'Off')",
        f"Preconditions:       original ExposureTime must be {EXPECTED_ORIGINAL_EXPOSURE_US!r} us and "
        f"{TARGET_EXPOSURE_US!r} us within the SDK limits",
        f"Exposure:            set_exposure_us({TARGET_EXPOSURE_US!r}), exact readback required",
        "Capture:             capture_frame() exactly once; validate and analyze in memory",
        "Close:               camera.close() always; the backend restores and verifies the original ExposureTime",
        "No retries. No files are saved. A forcibly killed process cannot restore the exposure.",
    ]


def _restoration_line(result):
    if not result.close_attempted:
        return "NOT ATTEMPTED; ExposureTime unverified"
    error = result.close_error
    if error is None:
        if result.restore_required is True:
            return (f"RESTORED_AND_VERIFIED by backend close() (original {result.original_exposure_us!r} us)")
        if result.restore_required is False:
            return "NOT_REQUIRED (no exposure write was issued)"
        return "close() succeeded (restoration verified or not required, per backend contract)"
    if isinstance(error, ic.IRaypleCloseError):
        restore = error.restore
        if error.restore_ok:
            return f"{restore.status} (cleanup failed separately)"
        detail = f"FAILED: {restore.status} (original {restore.original!r}"
        if restore.readback is not None:
            detail += f", read back {restore.readback!r}"
        return detail + "); check ExposureTime in MV Viewer"
    return "UNVERIFIED (close() did not complete normally); check ExposureTime in MV Viewer"


def _close_line(result):
    if not result.close_attempted:
        return "NOT ATTEMPTED"
    error = result.close_error
    if error is None:
        return "OK"
    if isinstance(error, ic.IRaypleCloseError):
        return f"FAILED: {error}"
    if isinstance(error, str):
        return f"FAILED: {error}"
    return f"FAILED: {error!r}"


def _value(value):
    return "not reached" if value is None else repr(value)


def format_result(result):
    lines = [
        f"Model:               {result.model or 'not reached'}",
        f"Serial (masked):     {result.serial or 'not reached'}",
        f"Original exposure:   {_value(result.original_exposure_us)} us",
        f"SDK exposure limits: {_value(result.limits_us)} us",
        f"Requested exposure:  {_value(result.requested_exposure_us)} us",
        f"Applied exposure:    {_value(result.applied_exposure_us)} us (backend strict readback)",
        f"capture_frame calls: {result.capture_calls}",
    ]
    if result.width is not None:
        lines.append(f"Frame:               {result.width} x {result.height}, dtype {result.dtype}, "
                     f"source={result.source}")
    if result.frame_id is not None:
        lines.append(f"frame_id:            {result.frame_id} (per-backend acquisition counter)")
        block = "not available" if result.native_block_id is None else result.native_block_id
        lines.append(f"Native SDK blockId:  {block} (camera-reported; distinct from frame_id)")
    if result.stats is not None:
        stats = result.stats
        lines.append(f"min DN {stats['min_dn']}, max DN {stats['max_dn']}, mean DN {stats['mean_dn']:.4f}, "
                     f"saturation fraction {stats['saturation_fraction']:.6f} (pixels == 255 DN)")
    lines.append(f"Exposure restoration: {_restoration_line(result)}")
    lines.append(f"Backend close:       {_close_line(result)}")
    if result.status != STATUS_COMPLETE:
        lines.append(f"SMOKE_FAILURE: {result.status}: {result.message}")
    if result.ok:
        lines.append("SMOKE_TEST_STATUS: PASS")
    else:
        lines.append("SMOKE_TEST_STATUS: FAILED")
    return lines


def exit_code_for(result):
    if result.ok:
        return EXIT_OK
    if result.interrupted:
        return EXIT_INTERRUPTED
    if not result.restore_ok:
        return EXIT_RESTORE_FAILED
    if not result.close_ok:
        return EXIT_CLEANUP_FAILED
    if result.status in PRECONDITION_STATUSES:
        return EXIT_PRECONDITION
    return EXIT_FAILED


def main(argv=None, input_fn=input, stdin=None):
    parser = argparse.ArgumentParser(
        description="One-frame smoke test of the IRaypleCamera backend on the iRAYPLE MV-A7200MU130."
    )
    parser.add_argument("--check-once", action="store_true",
                        help="required: run the single smoke test (interactive confirmation follows)")
    args = parser.parse_args(argv)

    print("LUMEN IRaypleCamera backend smoke test (one frame, one session)")
    if not args.check_once:
        print("NO_ACTION: pass --check-once to run the smoke test. No SDK was loaded and no camera was accessed.")
        return EXIT_REFUSED

    stdin = sys.stdin if stdin is None else stdin
    if not first_frame._is_interactive(stdin):
        print("REFUSED: an interactive terminal is required for confirmation. No camera was accessed.")
        return EXIT_REFUSED
    for line in format_plan():
        print(line)
    try:
        response = input_fn(f"Type {CONFIRMATION_PHRASE} to run this smoke test: ")
    except (EOFError, KeyboardInterrupt):
        print("\nREFUSED: no confirmation received. No camera was accessed.")
        return EXIT_REFUSED
    if response != CONFIRMATION_PHRASE:
        print("REFUSED: confirmation phrase did not match. No camera was accessed.")
        return EXIT_REFUSED

    try:
        sdk = ic.preload_vendor_sdk()
    except discovery.DiscoveryError as exc:
        print(f"SMOKE_FAILURE: SDK_LOAD_FAILED: {exc}")
        print("SMOKE_TEST_STATUS: FAILED")
        return EXIT_FAILED
    camera = ic.IRaypleCamera(sdk)

    try:
        result = run_smoke_test(camera)
    except KeyboardInterrupt:
        print("SMOKE_FAILURE: INTERRUPTED during close(); camera state and ExposureTime unknown.")
        print("SMOKE_TEST_STATUS: FAILED")
        return EXIT_INTERRUPTED
    for line in format_result(result):
        print(line)
    if result.stats is not None:
        print(SCOPE_NOTE)
    return exit_code_for(result)


if __name__ == "__main__":
    sys.exit(main())
