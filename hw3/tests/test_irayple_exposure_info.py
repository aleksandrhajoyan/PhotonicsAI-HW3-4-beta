import contextlib
import ctypes
import importlib
import io
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_exposure_info as ei  # noqa: E402

FULL_SERIAL = "DA1234567890"
FULL_CAMERA_KEY = "Huaray Technology:" + FULL_SERIAL
NAN = float("nan")
INF = float("inf")

READ_ONLY_CALLS = {
    "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_GetDoubleFeatureValue",
    "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax", "IMV_FeatureIsAvailable",
    "IMV_FeatureIsReadable", "IMV_GetEnumFeatureSymbol", "IMV_Close", "IMV_DestroyHandle",
}
FORBIDDEN_CALLS = (
    "IMV_SetDoubleFeatureValue", "IMV_SetIntFeatureValue", "IMV_SetBoolFeatureValue",
    "IMV_SetEnumFeatureValue", "IMV_SetEnumFeatureSymbol", "IMV_SetStringFeatureValue",
    "IMV_ExecuteCommandFeature", "IMV_StartGrabbing", "IMV_StartGrabbingEx", "IMV_GetFrame",
    "IMV_ReleaseFrame", "IMV_StopGrabbing", "IMV_AttachGrabbing", "IMV_OpenEx",
)


class FakeString(ctypes.Structure):
    """Same layout as the vendor IMV_String (IMVDefines.py / IMVDefines.h)."""

    _fields_ = [("str", ctypes.c_char * 256)]


def _device(model=b"MV-A7200MU130", serial=FULL_SERIAL.encode(), camera_type=1, interface_type=2):
    vendor = b"Huaray Technology"
    return SimpleNamespace(
        nCameraType=camera_type,
        nInterfaceType=interface_type,
        vendorName=vendor,
        modelName=model,
        serialNumber=serial,
        cameraKey=vendor + b":" + serial,
    )


class FakeSdk:
    """Fake vendor SDK. Every call goes into one ordered log; argument types follow the vendor wrapper."""

    def __init__(self, devices=None, codes=None, doubles=(10000.0, 15.0, 1000000.0),
                 enums=None, available=None, readable=None, raises=None):
        self.devices = [_device()] if devices is None else devices
        self.codes = {name: 0 for name in READ_ONLY_CALLS | set(FORBIDDEN_CALLS)}
        self.codes.update(codes or {})
        self.current, self.minimum, self.maximum = doubles
        self.enums = {"ExposureAuto": b"Off", "GainAuto": b"Off"} if enums is None else enums
        self.available = set(self.enums) if available is None else set(available)
        self.readable = set(self.available) if readable is None else set(readable)
        self.raises = raises or {}
        self.calls = []
        self.handles_alive = 0
        self.is_open = False
        self.ns = self._namespace()

    def names(self):
        return [call[0] for call in self.calls]

    def _call(self, name, *args):
        self.calls.append((name,) + args)
        if name in self.raises:
            raise self.raises[name]
        code = self.codes[name]
        if code == 0:
            getattr(self, "_on_" + name, lambda *a: None)(*args)
        return code

    def _on_IMV_EnumDevices(self, device_list, interface_type):
        device_list.nDevNum = len(self.devices)
        device_list.pDevInfo = list(self.devices)

    def _on_IMV_CreateHandle(self, mode, identifier):
        self.handles_alive += 1

    def _on_IMV_DestroyHandle(self):
        self.handles_alive -= 1

    def _on_IMV_Open(self):
        self.is_open = True

    def _on_IMV_Close(self):
        self.is_open = False

    def _write_double(self, name, out, value):
        assert name == "ExposureTime", name
        assert isinstance(out, ctypes.c_double), type(out)  # wrapper passes byref(pDoubleValue)
        out.value = value

    def _on_IMV_GetDoubleFeatureValue(self, name, out):
        self._write_double(name, out, self.current)

    def _on_IMV_GetDoubleFeatureMin(self, name, out):
        self._write_double(name, out, self.minimum)

    def _on_IMV_GetDoubleFeatureMax(self, name, out):
        self._write_double(name, out, self.maximum)

    def _on_IMV_GetEnumFeatureSymbol(self, name, symbol):
        assert isinstance(symbol, FakeString), type(symbol)  # wrapper passes byref(pEnumSymbol)
        symbol.str = self.enums[name]

    def _namespace(self):
        fake = self

        def method(name):
            return lambda self, *args: fake._call(name, *args)

        attrs = {name: method(name) for name in READ_ONLY_CALLS | set(FORBIDDEN_CALLS)}
        attrs["IMV_EnumDevices"] = staticmethod(lambda device_list, interface_type: fake._call(
            "IMV_EnumDevices", device_list, interface_type))

        def is_available(self, name):
            fake._call("IMV_FeatureIsAvailable", name)
            return name in fake.available

        def is_readable(self, name):
            fake._call("IMV_FeatureIsReadable", name)
            return name in fake.readable

        attrs["IMV_FeatureIsAvailable"] = is_available
        attrs["IMV_FeatureIsReadable"] = is_readable
        MvCamera = type("MvCamera", (), attrs)

        return SimpleNamespace(
            MvCamera=MvCamera,
            IMV_DeviceList=lambda: SimpleNamespace(nDevNum=0, pDevInfo=None),
            IMV_EInterfaceType=SimpleNamespace(
                interfaceTypeGige=0x1, interfaceTypeUsb3=0x2, interfaceTypeCL=0x4,
                interfaceTypePCIe=0x8, interfaceTypeAll=0x0,
            ),
            IMV_ECreateHandleMode=SimpleNamespace(
                modeByIndex=0, modeByCameraKey=1, modeByDeviceUserID=2, modeByIPAddress=3,
            ),
            IMV_String=FakeString,
            IMV_OK=0,
            IMV_ERROR=-101,
            IMV_INVALID_HANDLE=-102,
            IMV_INVALID_PARAM=-103,
            IMV_INVALID_ACCESS=-111,
            IMV_NOT_SUPPORT=-113,
            typeGigeCamera=0,
            typeU3vCamera=1,
            typeCLCamera=2,
            typePCIeCamera=3,
        )


def _run(fake=None, argv=("--inspect-exposure",), loader=None):
    if loader is None:
        loader = mock.Mock(return_value=fake.ns if fake else None)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ei.main(list(argv), sdk_loader=loader)
    return code, out.getvalue(), loader


class NoPhysicalSdkTestCase(unittest.TestCase):
    """Any attempt to load the real DLL or vendor wrapper during a test fails loudly."""

    def setUp(self):
        guard = AssertionError("real SDK access attempted in an offline test")
        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch.object(ei.discovery, "load_vendor_sdk", side_effect=guard),
        ):
            target.start()
            self.addCleanup(target.stop)

    def tearDown(self):
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


class ImportSafetyTests(unittest.TestCase):
    def test_import_performs_no_sdk_operations(self):
        with mock.patch("ctypes.WinDLL") as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir, \
                mock.patch("importlib.import_module") as import_module:
            importlib.reload(ei)
        for patched in (win_dll, cdll, add_dir, chdir, import_module):
            patched.assert_not_called()
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


class ManualExecutionSafetyTests(NoPhysicalSdkTestCase):
    def test_without_flag_no_camera_is_touched(self):
        fake = FakeSdk()
        code, out, loader = _run(fake, argv=())
        self.assertEqual(code, ei.EXIT_REFUSED)
        self.assertIn("No camera was accessed", out)
        loader.assert_not_called()
        self.assertEqual(fake.calls, [])

    def test_default_loader_not_used_without_flag(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ei.main([])
        self.assertEqual(code, ei.EXIT_REFUSED)
        ei.discovery.load_vendor_sdk.assert_not_called()

    def test_flag_loads_sdk_once(self):
        fake = FakeSdk()
        code, _, loader = _run(fake)
        self.assertEqual(code, ei.EXIT_OK)
        loader.assert_called_once_with()

    def test_sdk_load_failure_is_controlled(self):
        loader = mock.Mock(side_effect=ei.discovery.DiscoveryError("HuaRay SDK file not found: X"))
        code, out, _ = _run(loader=loader)
        self.assertEqual(code, ei.EXIT_QUERY_FAILED)
        self.assertIn("SDK_LOAD_FAILED", out)


class DeviceSelectionTests(NoPhysicalSdkTestCase):
    def test_exactly_one_usb3_enumeration(self):
        fake = FakeSdk()
        ei.inspect_exposure(fake.ns)
        enums = [call for call in fake.calls if call[0] == "IMV_EnumDevices"]
        self.assertEqual(len(enums), 1)
        self.assertEqual(enums[0][2], 0x2)  # interfaceTypeUsb3 only

    def test_handle_created_by_camera_key_then_opened(self):
        fake = FakeSdk()
        result = ei.inspect_exposure(fake.ns)
        self.assertTrue(result.ok)
        self.assertEqual(fake.names()[:3], ["IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open"])
        create_call = fake.calls[1]
        self.assertEqual(create_call[1], 1)  # modeByCameraKey
        self.assertIsInstance(create_call[2], ctypes.c_char_p)
        self.assertEqual(create_call[2].value, FULL_CAMERA_KEY.encode())
        self.assertEqual(fake.calls[2], ("IMV_Open",))  # IMV_Open(IN IMV_HANDLE) only

    def _assert_rejected_before_open(self, fake, status):
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_DEVICE_SELECTION)
        self.assertIn(f"INSPECTION_FAILED: {status}", out)
        self.assertEqual(fake.names(), ["IMV_EnumDevices"])
        self.assertNotIn("Real hardware device inspected", out)
        self.assertNotIn(FULL_SERIAL, out)

    def test_zero_devices_rejected(self):
        self._assert_rejected_before_open(FakeSdk(devices=[]), "NO_DEVICE")

    def test_multiple_devices_rejected(self):
        devices = [_device(), _device(serial=b"DA0000001111")]
        self._assert_rejected_before_open(FakeSdk(devices=devices), "MULTIPLE_DEVICES")

    def test_wrong_model_rejected(self):
        self._assert_rejected_before_open(FakeSdk(devices=[_device(model=b"MV-OTHER")]), "UNEXPECTED_DEVICE")

    def test_wrong_transport_rejected(self):
        self._assert_rejected_before_open(FakeSdk(devices=[_device(camera_type=0, interface_type=1)]),
                                          "UNEXPECTED_TRANSPORT")

    def test_identifiers_not_printed_in_full(self):
        _, out, _ = _run(FakeSdk())
        self.assertNotIn(FULL_SERIAL, out)
        self.assertNotIn(FULL_CAMERA_KEY, out)
        self.assertIn("****7890", out)


class ExposureReadTests(NoPhysicalSdkTestCase):
    def _double_calls(self, fake, name):
        return [call for call in fake.calls if call[0] == name]

    def test_current_minimum_and_maximum_are_read(self):
        fake = FakeSdk(doubles=(12345.5, 15.0, 2000000.0))
        result = ei.inspect_exposure(fake.ns)
        self.assertTrue(result.ok)
        self.assertEqual(result.exposure, ei.ExposureRange(12345.5, 15.0, 2000000.0))
        for name in ("IMV_GetDoubleFeatureValue", "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax"):
            with self.subTest(name=name):
                calls = self._double_calls(fake, name)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], "ExposureTime")
                self.assertIsInstance(calls[0][2], ctypes.c_double)

    def test_values_reported_without_unconfirmed_unit_claim(self):
        code, out, _ = _run(FakeSdk(doubles=(12345.5, 15.0, 2000000.0)))
        self.assertEqual(code, ei.EXIT_OK)
        self.assertIn("ExposureTime current (SDK): 12345.5", out)
        self.assertIn("ExposureTime minimum (SDK): 15.0", out)
        self.assertIn("ExposureTime maximum (SDK): 2000000.0", out)
        self.assertIn("ExposureTime unit: not confirmed", out)
        self.assertIsNone(re.search(r"\d\s*(us|µs|μs)\b", out))
        self.assertIn("Real hardware device inspected", out)
        self.assertIn("No camera settings changed", out)
        self.assertIn("EXPOSURE_STATUS: EXPOSURE_INFO_READ", out)

    def test_auto_modes_read_when_supported(self):
        fake = FakeSdk(enums={"ExposureAuto": b"Continuous", "GainAuto": b"Off"})
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_OK)
        reads = [call[1] for call in fake.calls if call[0] == "IMV_GetEnumFeatureSymbol"]
        self.assertEqual(reads, ["ExposureAuto", "GainAuto"])
        self.assertIn("ExposureAuto: Continuous", out)
        self.assertIn("GainAuto: Off", out)

    def test_unsupported_optional_features_reported_unavailable(self):
        fake = FakeSdk(enums={}, available=())
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_OK)
        self.assertNotIn("IMV_GetEnumFeatureSymbol", fake.names())
        self.assertIn("ExposureAuto: unavailable (device reports feature not available)", out)
        self.assertIn("GainAuto: unavailable (device reports feature not available)", out)
        self.assertNotIn("ExposureAuto: Off", out)
        self.assertNotIn("GainAuto: Off", out)

    def test_unreadable_or_failed_optional_reads_are_unavailable(self):
        fake = FakeSdk(readable={"GainAuto"}, codes={"IMV_GetEnumFeatureSymbol": -111})
        result = ei.inspect_exposure(fake.ns)
        self.assertTrue(result.ok)
        by_name = {feature.name: feature for feature in result.optional}
        self.assertFalse(by_name["ExposureAuto"].available)
        self.assertIn("not readable", by_name["ExposureAuto"].note)
        self.assertFalse(by_name["GainAuto"].available)
        self.assertIn("code -111 IMV_INVALID_ACCESS", by_name["GainAuto"].note)

    def test_empty_optional_symbol_is_unavailable(self):
        result = ei.inspect_exposure(FakeSdk(enums={"ExposureAuto": b"", "GainAuto": b"Off"}).ns)
        self.assertTrue(result.ok)
        self.assertFalse(result.optional[0].available)
        self.assertEqual(result.optional[1].value, "Off")


class ExposureValidationTests(NoPhysicalSdkTestCase):
    def _assert_rejected(self, doubles, status, fragment):
        fake = FakeSdk(doubles=doubles)
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_INVALID_VALUES)
        self.assertIn(f"INSPECTION_FAILED: {status}", out)
        self.assertIn(fragment, out)
        self.assertNotIn("EXPOSURE_STATUS", out)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.handles_alive, 0)

    def test_nan_and_infinite_values_rejected(self):
        for doubles in ((NAN, 15.0, 100.0), (50.0, NAN, 100.0), (50.0, 15.0, NAN),
                        (INF, 15.0, 100.0), (50.0, -INF, 100.0), (50.0, 15.0, INF)):
            with self.subTest(doubles=doubles):
                self._assert_rejected(doubles, ei.STATUS_INVALID_VALUE, "is not finite")

    def test_invalid_ranges_rejected(self):
        self._assert_rejected((0.0, -1.0, 100.0), ei.STATUS_INVALID_RANGE, "is negative")
        self._assert_rejected((50.0, 50.0, 50.0), ei.STATUS_INVALID_RANGE, "not greater than minimum")
        self._assert_rejected((50.0, 100.0, 15.0), ei.STATUS_INVALID_RANGE, "not greater than minimum")

    def test_out_of_range_current_rejected_without_normalizing(self):
        for current in (14.0, 100.5):
            with self.subTest(current=current):
                fake = FakeSdk(doubles=(current, 15.0, 100.0))
                result = ei.inspect_exposure(fake.ns)
                self.assertEqual(result.status, ei.STATUS_OUT_OF_RANGE)
                self.assertEqual(result.exposure.current, current)
                self._assert_rejected((current, 15.0, 100.0), ei.STATUS_OUT_OF_RANGE, "outside the reported range")

    def test_boundary_values_accepted(self):
        for current in (15.0, 100.0):
            with self.subTest(current=current):
                self.assertTrue(ei.inspect_exposure(FakeSdk(doubles=(current, 15.0, 100.0)).ns).ok)


class QueryFailureTests(NoPhysicalSdkTestCase):
    def test_required_query_failure_is_nonzero_and_keeps_code(self):
        for name in ("IMV_GetDoubleFeatureValue", "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax"):
            with self.subTest(name=name):
                fake = FakeSdk(codes={name: -111})
                code, out, _ = _run(fake)
                self.assertEqual(code, ei.EXIT_QUERY_FAILED)
                self.assertIn("INSPECTION_FAILED: EXPOSURE_QUERY_FAILED", out)
                self.assertIn("(error code -111 IMV_INVALID_ACCESS)", out)
                result = ei.inspect_exposure(FakeSdk(codes={name: -113}).ns)
                self.assertEqual(result.sdk_code, -113)
                self.assertEqual(result.sdk_code_name, "IMV_NOT_SUPPORT")
                self.assertIsNone(result.exposure)

    def test_query_failure_still_cleans_up(self):
        fake = FakeSdk(codes={"IMV_GetDoubleFeatureMax": -101})
        result = ei.inspect_exposure(fake.ns)
        self.assertFalse(result.ok)
        self.assertEqual([step.name for step in result.cleanup], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertTrue(result.cleanup_ok)
        self.assertFalse(fake.is_open)
        self.assertEqual(fake.handles_alive, 0)

    def test_query_exception_still_cleans_up(self):
        fake = FakeSdk(raises={"IMV_GetDoubleFeatureValue": OSError("access violation")})
        result = ei.inspect_exposure(fake.ns)
        self.assertEqual(result.status, ei.STATUS_UNEXPECTED_ERROR)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.handles_alive, 0)

    def test_keyboard_interrupt_during_query_cleans_up(self):
        fake = FakeSdk(raises={"IMV_GetDoubleFeatureMin": KeyboardInterrupt()})
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_INTERRUPTED)
        self.assertIn("INSPECTION_FAILED: INTERRUPTED", out)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.handles_alive, 0)

    def test_queries_are_not_retried(self):
        for name in ("IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_GetDoubleFeatureValue",
                     "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax"):
            with self.subTest(name=name):
                fake = FakeSdk(codes={name: -101})
                ei.inspect_exposure(fake.ns)
                self.assertEqual(fake.names().count(name), 1)

    def test_create_handle_failure_cleans_nothing(self):
        fake = FakeSdk(codes={"IMV_CreateHandle": -101})
        result = ei.inspect_exposure(fake.ns)
        self.assertEqual(result.status, "CREATE_HANDLE_FAILED")
        self.assertEqual(fake.names(), ["IMV_EnumDevices", "IMV_CreateHandle"])
        self.assertEqual(result.cleanup, [])

    def test_open_failure_destroys_handle_only(self):
        fake = FakeSdk(codes={"IMV_Open": -102})
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_QUERY_FAILED)
        self.assertIn("(error code -102 IMV_INVALID_HANDLE)", out)
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")
        self.assertNotIn("IMV_Close", fake.names())
        self.assertNotIn("Real hardware device inspected", out)
        self.assertEqual(fake.handles_alive, 0)


class CleanupTests(NoPhysicalSdkTestCase):
    def test_success_call_order_and_cleanup(self):
        fake = FakeSdk()
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_OK)
        self.assertEqual(fake.names(), [
            "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open",
            "IMV_GetDoubleFeatureValue", "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax",
            "IMV_FeatureIsAvailable", "IMV_FeatureIsReadable", "IMV_GetEnumFeatureSymbol",
            "IMV_FeatureIsAvailable", "IMV_FeatureIsReadable", "IMV_GetEnumFeatureSymbol",
            "IMV_Close", "IMV_DestroyHandle",
        ])
        self.assertIn("Cleanup IMV_Close: OK", out)
        self.assertIn("Cleanup IMV_DestroyHandle: OK", out)
        self.assertFalse(fake.is_open)
        self.assertEqual(fake.handles_alive, 0)

    def test_cleanup_failure_prevents_success(self):
        for name in ("IMV_Close", "IMV_DestroyHandle"):
            with self.subTest(name=name):
                fake = FakeSdk(codes={name: -101})
                code, out, _ = _run(fake)
                self.assertEqual(code, ei.EXIT_CLEANUP_FAILED)
                self.assertIn(f"Cleanup {name}: FAILED with code -101", out)
                self.assertIn("CLEANUP_FAILURE", out)
                self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")

    def test_cleanup_exception_is_visible_and_destroy_still_runs(self):
        fake = FakeSdk(raises={"IMV_Close": OSError("boom")})
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_CLEANUP_FAILED)
        self.assertIn("Cleanup IMV_Close: FAILED (OSError('boom'))", out)
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")
        self.assertEqual(fake.handles_alive, 0)

    def test_keyboard_interrupt_during_cleanup_continues_cleanup(self):
        fake = FakeSdk(raises={"IMV_Close": KeyboardInterrupt()})
        code, out, _ = _run(fake)
        self.assertEqual(code, ei.EXIT_INTERRUPTED)
        self.assertIn("Cleanup IMV_Close: FAILED (interrupted", out)
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")


class ReadOnlySafetyTests(NoPhysicalSdkTestCase):
    def _all_scenarios(self):
        return (FakeSdk(), FakeSdk(available=()), FakeSdk(doubles=(NAN, 15.0, 100.0)),
                FakeSdk(codes={"IMV_GetDoubleFeatureMin": -101}), FakeSdk(codes={"IMV_Close": -101}))

    def test_only_read_only_sdk_calls_are_made(self):
        for index, fake in enumerate(self._all_scenarios()):
            with self.subTest(scenario=index):
                _run(fake)
                self.assertTrue(set(fake.names()) <= READ_ONLY_CALLS)
                for forbidden in FORBIDDEN_CALLS:
                    self.assertNotIn(forbidden, fake.names())

    def test_only_exposure_related_features_are_queried(self):
        fake = FakeSdk()
        ei.inspect_exposure(fake.ns)
        feature_names = {call[1] for call in fake.calls
                         if call[0].startswith("IMV_GetDouble") or call[0].startswith("IMV_Feature")
                         or call[0] == "IMV_GetEnumFeatureSymbol"}
        self.assertEqual(feature_names, {"ExposureTime", "ExposureAuto", "GainAuto"})

    def test_no_openai_or_credential_access(self):
        had_openai = "openai" in sys.modules
        with mock.patch("os.getenv", side_effect=AssertionError("environment read")):
            code, _, _ = _run(FakeSdk())
        self.assertEqual(code, ei.EXIT_OK)
        if not had_openai:
            self.assertNotIn("openai", sys.modules)


class SourceSafetyTests(unittest.TestCase):
    def test_source_has_no_configuration_writes_or_extra_activity(self):
        source = (HW3_DIR / "irayple_exposure_info.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"IMV_Set\w*Feature", source))
        for forbidden in ("ExecuteCommand", "StartGrabbing", "GetFrame", "ReleaseFrame", "StopGrabbing",
                          "Trigger", "Reset", "interfaceTypeAll", "modeByIndex", "OpenEx", "threading",
                          "subprocess", "environ", "getenv", "open(", "SaveImage", "write_text",
                          "write_bytes"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("openai", source.lower())
        self.assertNotIn("api_key", source.lower())


if __name__ == "__main__":
    unittest.main()
