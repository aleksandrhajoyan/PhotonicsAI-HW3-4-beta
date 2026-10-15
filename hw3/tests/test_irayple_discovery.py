import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_discovery  # noqa: E402


def _discovery_error_type():
    # Looked up at call time: ImportSafetyTests reloads the module, which redefines the class.
    return irayple_discovery.DiscoveryError

FULL_SERIAL = "DA1234567890"


def _device(model=b"MV-A7200MU130", vendor=b"Huaray Technology", serial=FULL_SERIAL.encode(), camera_type=1, interface_type=2):
    """Shape of IMV_DeviceInfo fields as ctypes exposes them (c_char arrays read as bytes)."""
    return SimpleNamespace(
        nCameraType=camera_type,
        nInterfaceType=interface_type,
        vendorName=vendor,
        modelName=model,
        serialNumber=serial,
        cameraKey=vendor + b":" + serial,
    )


def _fake_sdk(devices=(), code=0):
    """Fake mirroring the attribute names of the vendor IMVApi/IMVDefines namespace."""
    camera = mock.MagicMock(name="MvCamera")

    def enum_devices(device_list, interface_type):
        if code == 0:
            device_list.nDevNum = len(devices)
            device_list.pDevInfo = list(devices)
        return code

    camera.IMV_EnumDevices.side_effect = enum_devices
    return SimpleNamespace(
        MvCamera=camera,
        IMV_DeviceList=lambda: SimpleNamespace(nDevNum=0, pDevInfo=None),
        IMV_EInterfaceType=SimpleNamespace(
            interfaceTypeGige=0x1, interfaceTypeUsb3=0x2, interfaceTypeCL=0x4,
            interfaceTypePCIe=0x8, interfaceTypeAll=0x0,
        ),
        IMV_OK=0,
        IMV_ERROR=-101,
        IMV_INVALID_PARAM=-103,
        typeGigeCamera=0,
        typeU3vCamera=1,
        typeCLCamera=2,
        typePCIeCamera=3,
    )


def _run(sdk=None, loader=None):
    loader = loader or (lambda: sdk)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = irayple_discovery.main([], sdk_loader=loader)
    return code, out.getvalue()


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_vendor_sdk(self):
        with mock.patch("ctypes.WinDLL") as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir:
            importlib.reload(irayple_discovery)
        win_dll.assert_not_called()
        cdll.assert_not_called()
        add_dir.assert_not_called()
        chdir.assert_not_called()
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)

    def test_import_does_not_enumerate(self):
        with mock.patch("importlib.import_module") as import_module:
            importlib.reload(irayple_discovery)
        import_module.assert_not_called()
        self.assertIsNone(irayple_discovery._dll_directory_handle)


class EnumerationTests(unittest.TestCase):
    def test_exactly_one_enumeration_per_invocation(self):
        sdk = _fake_sdk([_device()])
        loader = mock.Mock(return_value=sdk)
        _run(loader=loader)
        loader.assert_called_once_with()
        sdk.MvCamera.IMV_EnumDevices.assert_called_once()
        _run(loader=loader)
        self.assertEqual(sdk.MvCamera.IMV_EnumDevices.call_count, 2)

    def test_enumerates_usb3_interface_only(self):
        sdk = _fake_sdk([_device()])
        _run(sdk)
        (_, interface_type), _ = sdk.MvCamera.IMV_EnumDevices.call_args
        self.assertEqual(interface_type, 0x2)

    def test_zero_devices_is_nonzero(self):
        code, out = _run(_fake_sdk([]))
        self.assertEqual(code, irayple_discovery.EXIT_NO_DEVICES)
        self.assertIn("Devices discovered: 0", out)
        self.assertIn("NO_DEVICE_FOUND", out)

    def test_expected_camera_found(self):
        code, out = _run(_fake_sdk([_device()]))
        self.assertEqual(code, irayple_discovery.EXIT_OK)
        self.assertIn("Devices discovered: 1", out)
        self.assertIn("Model:     MV-A7200MU130", out)
        self.assertIn("Vendor:    Huaray Technology", out)
        self.assertIn("Expected MV-A7200MU130 found: YES (device index [1])", out)
        self.assertIn("DISCOVERY_OK", out)

    def test_multiple_devices_listed_without_selection(self):
        devices = [_device(model=b"MV-OTHER", serial=b"XX00001111"), _device()]
        sdk = _fake_sdk(devices)
        code, out = _run(sdk)
        self.assertEqual(code, irayple_discovery.EXIT_OK)
        self.assertIn("Devices discovered: 2", out)
        self.assertIn("[1] Model:     MV-OTHER", out)
        self.assertIn("[2] Model:     MV-A7200MU130", out)
        self.assertIn("device index [2]", out)
        self.assertIn("no device was selected", out)

    def test_other_device_only_is_nonzero(self):
        code, out = _run(_fake_sdk([_device(model=b"MV-OTHER")]))
        self.assertEqual(code, irayple_discovery.EXIT_EXPECTED_NOT_FOUND)
        self.assertIn("Expected MV-A7200MU130 found: NO", out)

    def test_model_decoded_from_sdk_bytes(self):
        record = irayple_discovery.parse_device(0, _device(model=b"MV-A7200MU130 "))
        self.assertEqual(record.model, "MV-A7200MU130")
        self.assertTrue(irayple_discovery.is_expected(record))
        undecodable = irayple_discovery.parse_device(0, _device(model=b"MV-\xff"))
        self.assertEqual(undecodable.model, "MV-�")

    def test_usb_transport_reported(self):
        _, out = _run(_fake_sdk([_device()]))
        self.assertIn("Transport: USB3 Vision (U3V) / interface USB3 [nCameraType=1, nInterfaceType=2]", out)

    def test_unknown_transport_reported_raw(self):
        _, out = _run(_fake_sdk([_device(camera_type=255, interface_type=99)]))
        self.assertIn("undefined (255)", out)
        self.assertIn("unknown (99)", out)

    def test_vendor_error_code_is_nonzero(self):
        code, out = _run(_fake_sdk([_device()], code=-101))
        self.assertEqual(code, irayple_discovery.EXIT_SDK_ERROR)
        self.assertIn("error code -101 (IMV_ERROR)", out)
        self.assertNotIn("DISCOVERY_OK", out)

    def test_enumerate_devices_raises_with_sdk_code(self):
        with self.assertRaises(_discovery_error_type()) as ctx:
            irayple_discovery.enumerate_devices(_fake_sdk(code=-103))
        self.assertEqual(ctx.exception.sdk_code, -103)


class MissingSdkTests(unittest.TestCase):
    def test_missing_sdk_files_raise_controlled_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("importlib.import_module") as import_module:
            with self.assertRaises(_discovery_error_type()) as ctx:
                irayple_discovery.load_vendor_sdk(Path(tmp) / "MVSDK", Path(tmp) / "x64")
        self.assertIn("HuaRay SDK file not found", str(ctx.exception))
        add_dir.assert_not_called()
        import_module.assert_not_called()

    def test_missing_sdk_reported_by_main(self):
        def loader():
            raise _discovery_error_type()("HuaRay SDK file not found: X")

        code, out = _run(loader=loader)
        self.assertEqual(code, irayple_discovery.EXIT_SDK_ERROR)
        self.assertIn("DISCOVERY_ERROR: HuaRay SDK file not found", out)


class HardwareSafetyTests(unittest.TestCase):
    def test_only_enumeration_is_called_on_sdk(self):
        sdk = _fake_sdk([_device(), _device(model=b"MV-OTHER")])
        _run(sdk)
        names = [name for name, _, _ in sdk.MvCamera.mock_calls]
        self.assertEqual(names, ["IMV_EnumDevices"])

    def test_no_handle_open_grab_or_feature_calls(self):
        sdk = _fake_sdk([_device()])
        _run(sdk)
        camera = sdk.MvCamera
        camera.assert_not_called()  # no MvCamera() instance, so no handle
        for forbidden in (
            "IMV_CreateHandle", "IMV_DestroyHandle", "IMV_Open", "IMV_OpenEx", "IMV_Close",
            "IMV_StartGrabbing", "IMV_StartGrabbingEx", "IMV_StopGrabbing", "IMV_AttachGrabbing",
            "IMV_GetFrame", "IMV_ReleaseFrame", "IMV_SetIntFeatureValue",
            "IMV_SetDoubleFeatureValue", "IMV_SetBoolFeatureValue", "IMV_SetEnumFeatureValue",
            "IMV_SetEnumFeatureSymbol", "IMV_SetStringFeatureValue", "IMV_ExecuteCommandFeature",
        ):
            getattr(camera, forbidden).assert_not_called()

    def test_source_has_no_forbidden_sdk_calls(self):
        source = (HW3_DIR / "irayple_discovery.py").read_text(encoding="utf-8")
        for forbidden in ("IMV_CreateHandle", "IMV_Open", "IMV_StartGrabbing", "IMV_GetFrame",
                          "FeatureValue", "FeatureSymbol", "ExecuteCommand", "interfaceTypeAll"):
            self.assertNotIn(forbidden, source)


class OutputTests(unittest.TestCase):
    def test_identifier_masked(self):
        _, out = _run(_fake_sdk([_device()]))
        self.assertNotIn(FULL_SERIAL, out)
        self.assertIn("Serial:    ****7890", out)

    def test_mask_identifier_edges(self):
        self.assertEqual(irayple_discovery.mask_identifier(""), "(not reported)")
        self.assertEqual(irayple_discovery.mask_identifier("abc"), "****")
        self.assertEqual(irayple_discovery.mask_identifier("ABCDEFGH"), "****EFGH")

    def test_output_separates_sdk_evidence_from_mv_viewer(self):
        _, out = _run(_fake_sdk([_device()]))
        self.assertIn("SDK enumeration evidence only", out)
        self.assertIn("separate from the MV Viewer", out)
        self.assertIn("no frames were acquired", out)


if __name__ == "__main__":
    unittest.main()
