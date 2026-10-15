"""Read-only inspection of the exposure features of the iRAYPLE MV-A7200MU130 via the official HuaRay IMV SDK.

Manual, single-shot utility. Nothing is loaded from the vendor SDK on import.
main() touches the camera only with --inspect-exposure: one USB3 enumeration,
CreateHandle (by camera key), Open, feature reads, Close, DestroyHandle. No
feature is ever written and no image acquisition is started, so exposure, gain,
pixel format and acquisition mode stay exactly as the camera reports them.

SDK facts this module relies on (Development/Include/IMV/IMVApi.h, IMVDefines.h,
Samples/Python/IMV/MVSDK/IMVApi.py):
    int  IMV_GetDoubleFeatureValue(IN IMV_HANDLE, IN const char* pFeatureName, OUT double* pDoubleValue)
    int  IMV_GetDoubleFeatureMin  (IN IMV_HANDLE, IN const char* pFeatureName, OUT double* pDoubleValue)
    int  IMV_GetDoubleFeatureMax  (IN IMV_HANDLE, IN const char* pFeatureName, OUT double* pDoubleValue)
    int  IMV_GetEnumFeatureSymbol (IN IMV_HANDLE, IN const char* pFeatureName, OUT IMV_String* pEnumSymbol)
    bool IMV_FeatureIsAvailable   (IN IMV_HANDLE, IN const char* pFeatureName)
    bool IMV_FeatureIsReadable    (IN IMV_HANDLE, IN const char* pFeatureName)
    IMV_String is { char str[IMV_MAX_STRING_LENTH = 256]; }.
    The Python wrapper takes the feature name as str (it encodes UTF-8 itself) and
    passes the output object with byref(), so callers supply a ctypes.c_double or
    an IMV_String instance. The wrapper assigns ".argtype" (not ".argtypes"), so
    ctypes uses default argument conversion; restype is set correctly.

ExposureTime is read with the double getters, as in the vendor CommPropAccess
samples. The installed IMV headers and Python bindings do not state its unit, so
no unit is claimed. ExposureAuto and GainAuto do not appear in any installed
vendor text file; they are read only if the device reports them available and
readable, and otherwise reported as unavailable.
"""

import argparse
import ctypes
import math
import sys
from dataclasses import dataclass, field

import irayple_discovery as discovery
import irayple_first_frame as first_frame

EXPECTED_MODEL = discovery.EXPECTED_MODEL
EXPOSURE_FEATURE = "ExposureTime"
OPTIONAL_ENUM_FEATURES = ("ExposureAuto", "GainAuto")
EXPOSURE_UNIT_NOTE = (
    "not confirmed by the installed IMV SDK headers or bindings "
    "(GenICam SFNC convention is microseconds; treat as unverified)"
)

EXIT_OK = 0
EXIT_QUERY_FAILED = 1
EXIT_REFUSED = 2
EXIT_DEVICE_SELECTION = 3
EXIT_CLEANUP_FAILED = 4
EXIT_INVALID_VALUES = 5
EXIT_INTERRUPTED = 130

STATUS_READ = "EXPOSURE_INFO_READ"
STATUS_INVALID_VALUE = "INVALID_EXPOSURE_VALUE"
STATUS_INVALID_RANGE = "INVALID_EXPOSURE_RANGE"
STATUS_OUT_OF_RANGE = "EXPOSURE_OUT_OF_RANGE"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"

VALUE_STATUSES = (STATUS_INVALID_VALUE, STATUS_INVALID_RANGE, STATUS_OUT_OF_RANGE)


class InspectionError(first_frame.CaptureError):
    """Controlled inspection failure, tagged with a status and optional native code."""


@dataclass(frozen=True)
class ExposureRange:
    current: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class OptionalFeature:
    name: str
    value: str = None
    note: str = None

    @property
    def available(self):
        return self.value is not None


@dataclass
class InspectionResult:
    status: str
    message: str = ""
    sdk_code: int = None
    sdk_code_name: str = None
    device: first_frame.SelectedDevice = None
    opened: bool = False
    exposure: ExposureRange = None
    optional: list = field(default_factory=list)
    cleanup: list = field(default_factory=list)

    @property
    def cleanup_ok(self):
        return all(step.ok for step in self.cleanup)

    @property
    def ok(self):
        return self.status == STATUS_READ and self.exposure is not None and self.cleanup_ok


def validate_exposure(exposure):
    """Reject non-finite values and inconsistent ranges. Never clamps or normalizes."""
    for label, value in (("current", exposure.current), ("minimum", exposure.minimum),
                         ("maximum", exposure.maximum)):
        if not math.isfinite(value):
            raise InspectionError(STATUS_INVALID_VALUE, f"{EXPOSURE_FEATURE} {label} is not finite ({value!r})")
    if exposure.minimum < 0:
        raise InspectionError(STATUS_INVALID_RANGE, f"{EXPOSURE_FEATURE} minimum {exposure.minimum!r} is negative")
    if exposure.maximum <= exposure.minimum:
        raise InspectionError(
            STATUS_INVALID_RANGE,
            f"{EXPOSURE_FEATURE} maximum {exposure.maximum!r} is not greater than minimum {exposure.minimum!r}",
        )
    if not exposure.minimum <= exposure.current <= exposure.maximum:
        raise InspectionError(
            STATUS_OUT_OF_RANGE,
            f"{EXPOSURE_FEATURE} current {exposure.current!r} is outside the reported range "
            f"[{exposure.minimum!r}, {exposure.maximum!r}]",
        )


class _Session:
    """Tracks which native resources were acquired so cleanup releases exactly those, in reverse order."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.camera = None
        self.handle_created = False
        self.opened = False
        self.interrupted = False
        self.steps = []

    def _check(self, status, name, code):
        if code != self.sdk.IMV_OK:
            raise InspectionError(status, f"{name} failed", sdk_code=code)

    def read_double(self, getter_name):
        value = ctypes.c_double(0.0)
        code = getattr(self.camera, getter_name)(EXPOSURE_FEATURE, value)
        self._check("EXPOSURE_QUERY_FAILED", f"{getter_name}({EXPOSURE_FEATURE})", code)
        return float(value.value)

    def read_optional_enum(self, name):
        if not self.camera.IMV_FeatureIsAvailable(name):
            return OptionalFeature(name, note="unavailable (device reports feature not available)")
        if not self.camera.IMV_FeatureIsReadable(name):
            return OptionalFeature(name, note="unavailable (device reports feature not readable)")
        symbol = self.sdk.IMV_String()
        code = self.camera.IMV_GetEnumFeatureSymbol(name, symbol)
        if code != self.sdk.IMV_OK:
            label = first_frame.error_label(self.sdk, code)
            return OptionalFeature(name, note=f"unavailable (read failed with code {code} {label})")
        value = discovery.decode_field(symbol.str)
        if not value:
            return OptionalFeature(name, note="unavailable (SDK returned an empty symbol)")
        return OptionalFeature(name, value=value)

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
        result.opened = True

        exposure = ExposureRange(
            current=self.read_double("IMV_GetDoubleFeatureValue"),
            minimum=self.read_double("IMV_GetDoubleFeatureMin"),
            maximum=self.read_double("IMV_GetDoubleFeatureMax"),
        )
        result.exposure = exposure
        result.optional = [self.read_optional_enum(name) for name in OPTIONAL_ENUM_FEATURES]
        validate_exposure(exposure)
        result.status = STATUS_READ

    def _cleanup_call(self, name, call):
        try:
            code = call()
        except KeyboardInterrupt:
            self.interrupted = True
            self.steps.append(first_frame.CleanupStep(name, error="interrupted by KeyboardInterrupt"))
        except Exception as exc:  # native wrapper failure must not skip the remaining cleanup
            self.steps.append(first_frame.CleanupStep(name, error=repr(exc)))
        else:
            self.steps.append(first_frame.CleanupStep(name, code=code))

    def cleanup(self):
        if self.opened:
            self.opened = False
            self._cleanup_call("IMV_Close", self.camera.IMV_Close)
        if self.handle_created:
            self.handle_created = False
            self._cleanup_call("IMV_DestroyHandle", self.camera.IMV_DestroyHandle)
        return list(self.steps)


def inspect_exposure(sdk):
    """Run one enumerate/open/read/close cycle. Never raises for SDK failures.

    Only a KeyboardInterrupt that escapes cleanup itself can propagate.
    """
    session = _Session(sdk)
    result = InspectionResult(status=STATUS_UNEXPECTED_ERROR)
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
        result.message = "interrupted before inspection completed"
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
        lines.append(f"Model:                {result.device.model}")
        lines.append(f"Serial:               {result.device.masked_serial}")
    if result.opened:
        lines.append("Real hardware device inspected (opened read-only via IMV_Open).")
    exposure = result.exposure
    if exposure is not None:
        lines.extend(
            [
                f"{EXPOSURE_FEATURE} current (SDK): {exposure.current!r}",
                f"{EXPOSURE_FEATURE} minimum (SDK): {exposure.minimum!r}",
                f"{EXPOSURE_FEATURE} maximum (SDK): {exposure.maximum!r}",
                f"{EXPOSURE_FEATURE} unit: {EXPOSURE_UNIT_NOTE}",
            ]
        )
    for feature in result.optional:
        lines.append(f"{feature.name}: {feature.value if feature.available else feature.note}")
    if result.opened:
        lines.append("No camera settings changed (no feature write was issued).")
    for step in result.cleanup:
        if step.ok:
            lines.append(f"Cleanup {step.name}: OK")
        elif step.error is not None:
            lines.append(f"Cleanup {step.name}: FAILED ({step.error})")
        else:
            lines.append(f"Cleanup {step.name}: FAILED with code {step.code}")

    if result.ok:
        lines.append(f"EXPOSURE_STATUS: {STATUS_READ}")
    else:
        if result.status == STATUS_READ:
            lines.append(f"EXPOSURE_STATUS: {STATUS_READ}")
        else:
            detail = result.message
            if result.sdk_code is not None:
                detail += f" (error code {result.sdk_code} {result.sdk_code_name})"
            lines.append(f"INSPECTION_FAILED: {result.status}: {detail}")
        if not result.cleanup_ok:
            lines.append("CLEANUP_FAILURE: one or more SDK shutdown calls failed (see above)")
    return lines


def exit_code_for(result):
    if result.ok:
        return EXIT_OK
    if result.status == STATUS_INTERRUPTED:
        return EXIT_INTERRUPTED
    if not result.cleanup_ok:
        return EXIT_CLEANUP_FAILED
    if result.status in first_frame.DEVICE_SELECTION_STATUSES:
        return EXIT_DEVICE_SELECTION
    if result.status in VALUE_STATUSES:
        return EXIT_INVALID_VALUES
    return EXIT_QUERY_FAILED


def main(argv=None, sdk_loader=None):
    parser = argparse.ArgumentParser(
        description="Read the exposure features of the iRAYPLE MV-A7200MU130 (read-only, no acquisition)."
    )
    parser.add_argument(
        "--inspect-exposure",
        action="store_true",
        help="required: open the camera once and read ExposureTime, ExposureAuto and GainAuto",
    )
    args = parser.parse_args(argv)

    print("LUMEN iRAYPLE exposure inspection (read-only)")
    if not args.inspect_exposure:
        print("NO_ACTION: pass --inspect-exposure to read exposure features. No camera was accessed.")
        return EXIT_REFUSED

    loader = sdk_loader if sdk_loader is not None else discovery.load_vendor_sdk
    try:
        sdk = loader()
    except discovery.DiscoveryError as exc:
        print(f"INSPECTION_FAILED: SDK_LOAD_FAILED: {exc}")
        return EXIT_QUERY_FAILED

    try:
        result = inspect_exposure(sdk)
    except KeyboardInterrupt:
        print("INSPECTION_FAILED: INTERRUPTED during cleanup; camera state unknown.")
        return EXIT_INTERRUPTED
    for line in format_result(result):
        print(line)
    return exit_code_for(result)


if __name__ == "__main__":
    sys.exit(main())
