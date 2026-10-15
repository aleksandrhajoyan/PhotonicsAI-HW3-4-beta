"""Enumerate HuaRay / iRAYPLE cameras through the official installed IMV SDK.

Performs exactly one IMV_EnumDevices call when main() is invoked. Never creates
a device handle, opens a camera, starts grabbing, captures frames, or reads or
writes camera features. Importing this module loads nothing from the vendor SDK.

Enumeration is restricted to the USB3 Vision interface so that no GigE
discovery broadcast is sent on the local network.
"""

import argparse
import importlib
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_MODEL = "MV-A7200MU130"
DLL_NAME = "MVSDKmd.dll"
VENDOR_ROOT = Path(r"C:\Program Files\HuarayTech\MV Viewer")
VENDOR_RUNTIME_DIR = VENDOR_ROOT / "Runtime" / "x64"
VENDOR_MVSDK_DIR = VENDOR_ROOT / "Development" / "Samples" / "Python" / "IMV" / "MVSDK"

EXIT_OK = 0
EXIT_SDK_ERROR = 1
EXIT_NO_DEVICES = 2
EXIT_EXPECTED_NOT_FOUND = 3

# Return-code names defined in the vendor IMVDefines.py.
SDK_ERROR_NAMES = (
    "IMV_ERROR",
    "IMV_INVALID_HANDLE",
    "IMV_INVALID_PARAM",
    "IMV_INVALID_FRAME_HANDLE",
    "IMV_INVALID_FRAME",
    "IMV_INVALID_RESOURCE",
    "IMV_INVALID_IP",
    "IMV_NO_MEMORY",
    "IMV_INSUFFICIENT_MEMORY",
    "IMV_ERROR_PROPERTY_TYPE",
    "IMV_INVALID_ACCESS",
    "IMV_INVALID_RANGE",
    "IMV_NOT_SUPPORT",
)

EVIDENCE_NOTE = (
    "Evidence source: official HuaRay IMV SDK (IMV_EnumDevices via Python ctypes). "
    "This is SDK enumeration evidence only. It is separate from the MV Viewer "
    "live-view observation; no camera was opened and no frames were acquired."
)

# Keeps the os.add_dll_directory registration alive for the process lifetime.
_dll_directory_handle = None


class DiscoveryError(Exception):
    """Controlled discovery failure with an optional native SDK return code."""

    def __init__(self, message, sdk_code=None):
        super().__init__(message)
        self.sdk_code = sdk_code


@dataclass(frozen=True)
class DeviceRecord:
    index: int
    vendor: str
    model: str
    camera_type: int
    interface_type: int
    serial_number: str


def load_vendor_sdk(mvsdk_dir=VENDOR_MVSDK_DIR, runtime_dir=VENDOR_RUNTIME_DIR):
    """Import the vendor IMVApi wrapper, which loads MVSDKmd.dll on import.

    The vendor wrapper loads the DLL from a path relative to the current working
    directory ("../../../../../Runtime/x64/MVSDKmd.dll"), as its own samples do
    when run from inside the Samples/Python/IMV tree. The working directory is
    switched to the MVSDK folder only for the import and restored afterwards.
    """
    global _dll_directory_handle
    mvsdk_dir = Path(mvsdk_dir)
    runtime_dir = Path(runtime_dir)
    for required in (mvsdk_dir / "IMVApi.py", mvsdk_dir / "IMVDefines.py", runtime_dir / DLL_NAME):
        if not required.is_file():
            raise DiscoveryError(f"HuaRay SDK file not found: {required}")
    if sys.platform != "win32":
        raise DiscoveryError(f"Windows required, got platform {sys.platform!r}")
    if struct.calcsize("P") * 8 != 64:
        raise DiscoveryError("64-bit Python required for the x64 SDK runtime")

    if _dll_directory_handle is None:
        _dll_directory_handle = os.add_dll_directory(str(runtime_dir))

    previous_cwd = os.getcwd()
    sys.path.insert(0, str(mvsdk_dir))
    try:
        os.chdir(mvsdk_dir)
        return importlib.import_module("IMVApi")
    except (ImportError, OSError) as exc:
        raise DiscoveryError(f"Could not load HuaRay SDK wrapper: {exc!r}") from exc
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(str(mvsdk_dir))


def decode_field(raw):
    """Decode a ctypes c_char array field (bytes up to the first NUL)."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def mask_identifier(value):
    """Show only the last four characters of a device identifier."""
    if not value:
        return "(not reported)"
    if len(value) <= 4:
        return "****"
    return "****" + value[-4:]


def camera_type_label(sdk, camera_type):
    labels = {
        sdk.typeGigeCamera: "GigE Vision",
        sdk.typeU3vCamera: "USB3 Vision (U3V)",
        sdk.typeCLCamera: "Camera Link",
        sdk.typePCIeCamera: "PCIe",
    }
    return labels.get(camera_type, f"undefined ({camera_type})")


def interface_type_label(sdk, interface_type):
    interfaces = sdk.IMV_EInterfaceType
    labels = {
        interfaces.interfaceTypeGige: "GigE",
        interfaces.interfaceTypeUsb3: "USB3",
        interfaces.interfaceTypeCL: "Camera Link",
        interfaces.interfaceTypePCIe: "PCIe",
    }
    return labels.get(interface_type, f"unknown ({interface_type})")


def error_name(sdk, code):
    for name in SDK_ERROR_NAMES:
        if getattr(sdk, name, None) == code:
            return name
    return "unrecognized"


def parse_device(index, info):
    return DeviceRecord(
        index=index,
        vendor=decode_field(info.vendorName),
        model=decode_field(info.modelName),
        camera_type=int(info.nCameraType),
        interface_type=int(info.nInterfaceType),
        serial_number=decode_field(info.serialNumber),
    )


def enumerate_devices(sdk):
    """Call IMV_EnumDevices exactly once on the USB3 interface and parse results."""
    device_list = sdk.IMV_DeviceList()
    code = sdk.MvCamera.IMV_EnumDevices(device_list, sdk.IMV_EInterfaceType.interfaceTypeUsb3)
    if code != sdk.IMV_OK:
        raise DiscoveryError(
            f"IMV_EnumDevices failed with error code {code} ({error_name(sdk, code)})",
            sdk_code=code,
        )
    return [parse_device(i, device_list.pDevInfo[i]) for i in range(int(device_list.nDevNum))]


def is_expected(record):
    return record.model.upper() == EXPECTED_MODEL.upper()


def format_report(sdk, records):
    lines = [
        "IMV_EnumDevices result: IMV_OK (interface filter: USB3)",
        f"Devices discovered: {len(records)}",
    ]
    for record in records:
        lines.extend(
            [
                f"[{record.index + 1}] Model:     {record.model or '(not reported)'}",
                f"    Vendor:    {record.vendor or '(not reported)'}",
                f"    Transport: {camera_type_label(sdk, record.camera_type)}"
                f" / interface {interface_type_label(sdk, record.interface_type)}"
                f" [nCameraType={record.camera_type}, nInterfaceType={record.interface_type}]",
                f"    Serial:    {mask_identifier(record.serial_number)}",
            ]
        )
    matches = [record.index + 1 for record in records if is_expected(record)]
    if matches:
        lines.append(f"Expected {EXPECTED_MODEL} found: YES (device index {matches})")
        if len(records) > 1:
            lines.append("Multiple devices present; no device was selected.")
    else:
        lines.append(f"Expected {EXPECTED_MODEL} found: NO")
    return lines


def main(argv=None, sdk_loader=load_vendor_sdk):
    argparse.ArgumentParser(
        description="Enumerate iRAYPLE cameras once via the official HuaRay IMV SDK (no open, no capture)."
    ).parse_args(argv)

    print("LUMEN iRAYPLE discovery (enumeration only)")
    print(EVIDENCE_NOTE)
    try:
        sdk = sdk_loader()
        records = enumerate_devices(sdk)
    except DiscoveryError as exc:
        print(f"DISCOVERY_ERROR: {exc}")
        return EXIT_SDK_ERROR

    if not records:
        print("IMV_EnumDevices result: IMV_OK (interface filter: USB3)")
        print("Devices discovered: 0")
        print("NO_DEVICE_FOUND: the SDK reported no USB3 Vision cameras.")
        return EXIT_NO_DEVICES

    for line in format_report(sdk, records):
        print(line)
    if any(is_expected(record) for record in records):
        print("DISCOVERY_OK")
        return EXIT_OK
    return EXIT_EXPECTED_NOT_FOUND


if __name__ == "__main__":
    sys.exit(main())
