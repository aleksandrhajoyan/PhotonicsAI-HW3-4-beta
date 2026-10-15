import contextlib
import ctypes
import hashlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_first_frame as ff  # noqa: E402

FULL_SERIAL = "DA1234567890"
MONO8 = 0x01080001
MONO10 = 0x01100003
MONO12_PACKED = 0x010C0006
UNKNOWN_FORMAT = 0x01080099
MONO8_PAYLOAD = bytes([0, 10, 20, 30, 40, 50, 60, 255])  # 4 x 2


class FakeFrameInfo(ctypes.Structure):
    """Same layout as the vendor IMV_FrameInfo (IMVDefines.py / IMVDefines.h)."""

    _fields_ = [
        ("blockId", ctypes.c_uint64),
        ("status", ctypes.c_uint),
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("size", ctypes.c_uint),
        ("pixelFormat", ctypes.c_int),
        ("timeStamp", ctypes.c_uint64),
        ("chunkCount", ctypes.c_uint),
        ("paddingX", ctypes.c_uint),
        ("paddingY", ctypes.c_uint),
        ("recvFrameTime", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 19),
    ]


class FakeFrame(ctypes.Structure):
    _fields_ = [
        ("frameHandle", ctypes.c_void_p),
        ("pData", ctypes.POINTER(ctypes.c_ubyte)),
        ("frameInfo", FakeFrameInfo),
        ("nReserved", ctypes.c_uint * 10),
    ]


class FakeString(ctypes.Structure):
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
    """Fake vendor SDK. Every call goes into one ordered log; frame memory is fake 'SDK-owned' storage."""

    def __init__(self, devices=None, enum_code=0, create_code=0, open_code=0, trigger=b"Off",
                 trigger_code=0, start_code=0, get_frame_code=0, payload=MONO8_PAYLOAD, width=4,
                 height=2, pixel_format=MONO8, size=None, status=0, padding=(0, 0), null_data=False,
                 release_code=0, stop_code=0, close_code=0, destroy_code=0, raises=None):
        self.devices = [_device()] if devices is None else devices
        self.codes = {
            "IMV_EnumDevices": enum_code, "IMV_CreateHandle": create_code, "IMV_Open": open_code,
            "IMV_GetEnumFeatureSymbol": trigger_code, "IMV_StartGrabbing": start_code,
            "IMV_GetFrame": get_frame_code, "IMV_ReleaseFrame": release_code,
            "IMV_StopGrabbing": stop_code, "IMV_Close": close_code, "IMV_DestroyHandle": destroy_code,
        }
        self.trigger = trigger
        self.payload = payload
        self.width, self.height = width, height
        self.pixel_format = pixel_format
        self.size = len(payload) if size is None else size
        self.status = status
        self.padding = padding
        self.null_data = null_data
        self.raises = raises or {}
        self.calls = []
        self.handles_alive = 0
        self.is_open = False
        self.is_grabbing = False
        self.frames_outstanding = 0
        self.native_buffer = None
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

    def _on_IMV_StartGrabbing(self):
        self.is_grabbing = True

    def _on_IMV_StopGrabbing(self):
        self.is_grabbing = False

    def _on_IMV_GetEnumFeatureSymbol(self, name, symbol):
        symbol.str = self.trigger

    def _on_IMV_GetFrame(self, frame, timeout_ms):
        self.native_buffer = (ctypes.c_ubyte * len(self.payload)).from_buffer_copy(self.payload)
        if not self.null_data:
            frame.pData = ctypes.cast(self.native_buffer, ctypes.POINTER(ctypes.c_ubyte))
        info = frame.frameInfo
        info.blockId = 42
        info.status = self.status
        info.width, info.height = self.width, self.height
        info.size = self.size
        info.pixelFormat = self.pixel_format
        info.paddingX, info.paddingY = self.padding
        self.frames_outstanding += 1

    def _on_IMV_ReleaseFrame(self, frame):
        self.frames_outstanding -= 1
        # The SDK may reuse its buffer after release; overwrite it to prove nothing still points here.
        ctypes.memset(self.native_buffer, 0xEE, len(self.payload))

    def _namespace(self):
        fake = self

        class MvCamera:
            @staticmethod
            def IMV_EnumDevices(device_list, interface_type):
                return fake._call("IMV_EnumDevices", device_list, interface_type)

            def IMV_CreateHandle(self, mode, identifier):
                return fake._call("IMV_CreateHandle", mode, identifier)

            def IMV_Open(self):
                return fake._call("IMV_Open")

            def IMV_GetEnumFeatureSymbol(self, name, symbol):
                return fake._call("IMV_GetEnumFeatureSymbol", name, symbol)

            def IMV_StartGrabbing(self):
                return fake._call("IMV_StartGrabbing")

            def IMV_GetFrame(self, frame, timeout_ms):
                return fake._call("IMV_GetFrame", frame, timeout_ms)

            def IMV_ReleaseFrame(self, frame):
                return fake._call("IMV_ReleaseFrame", frame)

            def IMV_StopGrabbing(self):
                return fake._call("IMV_StopGrabbing")

            def IMV_Close(self):
                return fake._call("IMV_Close")

            def IMV_DestroyHandle(self):
                return fake._call("IMV_DestroyHandle")

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
            IMV_EPixelType=SimpleNamespace(
                gvspPixelTypeUndefined=-1, gvspPixelMono8=MONO8, gvspPixelMono10=MONO10,
                gvspPixelMono12Packed=MONO12_PACKED,
            ),
            IMV_Frame=FakeFrame,
            IMV_String=FakeString,
            IMV_OK=0,
            IMV_ERROR=-101,
            IMV_INVALID_HANDLE=-102,
            IMV_INVALID_PARAM=-103,
            typeGigeCamera=0,
            typeU3vCamera=1,
            typeCLCamera=2,
            typePCIeCamera=3,
        )


def _run(fake=None, argv=("--capture-once",), response=ff.CONFIRMATION_PHRASE, interactive=True,
         loader=None, input_fn=None):
    if loader is None:
        loader = mock.Mock(return_value=fake.ns if fake else None)
    if input_fn is None:
        input_fn = mock.Mock(return_value=response)
    stdin = SimpleNamespace(isatty=lambda: interactive)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ff.main(list(argv), sdk_loader=loader, input_fn=input_fn, stdin=stdin)
    return code, out.getvalue(), loader, input_fn


class NoPhysicalSdkTestCase(unittest.TestCase):
    """Any attempt to load the real DLL or vendor wrapper during a test fails loudly."""

    def setUp(self):
        guard = AssertionError("real SDK access attempted in an offline test")
        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch.object(ff.discovery, "load_vendor_sdk", side_effect=guard),
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
            importlib.reload(ff)
        for patched in (win_dll, cdll, add_dir, chdir, import_module):
            patched.assert_not_called()
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


class ManualExecutionSafetyTests(NoPhysicalSdkTestCase):
    def test_without_capture_once_no_camera_is_touched(self):
        fake = FakeSdk()
        code, out, loader, input_fn = _run(fake, argv=())
        self.assertEqual(code, ff.EXIT_REFUSED)
        self.assertIn("No camera was accessed", out)
        loader.assert_not_called()
        input_fn.assert_not_called()
        self.assertEqual(fake.calls, [])

    def test_default_loader_not_used_without_flag(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ff.main([], input_fn=mock.Mock(), stdin=SimpleNamespace(isatty=lambda: True))
        self.assertEqual(code, ff.EXIT_REFUSED)
        ff.discovery.load_vendor_sdk.assert_not_called()

    def test_noninteractive_terminal_refuses(self):
        fake = FakeSdk()
        code, out, loader, input_fn = _run(fake, interactive=False)
        self.assertEqual(code, ff.EXIT_REFUSED)
        self.assertIn("interactive terminal is required", out)
        input_fn.assert_not_called()
        loader.assert_not_called()
        self.assertEqual(fake.calls, [])

    def test_stdin_without_isatty_refuses(self):
        loader = mock.Mock()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ff.main(["--capture-once"], sdk_loader=loader, input_fn=mock.Mock(), stdin=object())
        self.assertEqual(code, ff.EXIT_REFUSED)
        loader.assert_not_called()

    def test_incorrect_confirmation_refuses(self):
        for response in ("", "y", "yes", "capture one frame", "CAPTURE ONE FRAME ", " CAPTURE ONE FRAME",
                         "CAPTURE  ONE FRAME", "CAPTURE ONE FRAMES"):
            with self.subTest(response=response):
                fake = FakeSdk()
                code, out, loader, _ = _run(fake, response=response)
                self.assertEqual(code, ff.EXIT_REFUSED)
                self.assertIn("did not match", out)
                loader.assert_not_called()
                self.assertEqual(fake.calls, [])

    def test_eof_and_keyboard_interrupt_at_prompt_refuse(self):
        for exc in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                fake = FakeSdk()
                code, out, loader, _ = _run(fake, input_fn=mock.Mock(side_effect=exc))
                self.assertEqual(code, ff.EXIT_REFUSED)
                self.assertIn("REFUSED", out)
                loader.assert_not_called()
                self.assertEqual(fake.calls, [])

    def test_exact_phrase_proceeds(self):
        fake = FakeSdk()
        code, _, loader, input_fn = _run(fake)
        self.assertEqual(code, ff.EXIT_OK)
        input_fn.assert_called_once()
        loader.assert_called_once_with()

    def test_sdk_load_failure_is_controlled(self):
        loader = mock.Mock(side_effect=ff.discovery.DiscoveryError("HuaRay SDK file not found: X"))
        code, out, _, _ = _run(loader=loader)
        self.assertEqual(code, ff.EXIT_CAPTURE_FAILED)
        self.assertIn("SDK_LOAD_FAILED", out)


class DeviceSelectionTests(NoPhysicalSdkTestCase):
    def test_single_expected_device_selected_by_camera_key(self):
        fake = FakeSdk()
        result = ff.capture_one_frame(fake.ns)
        self.assertTrue(result.ok)
        enum_call = fake.calls[0]
        self.assertEqual(enum_call[0], "IMV_EnumDevices")
        self.assertEqual(enum_call[2], 0x2)  # USB3 only
        create_call = fake.calls[1]
        self.assertEqual(create_call[0], "IMV_CreateHandle")
        self.assertEqual(create_call[1], 1)  # modeByCameraKey
        self.assertIsInstance(create_call[2], ctypes.c_char_p)
        self.assertEqual(create_call[2].value, b"Huaray Technology:" + FULL_SERIAL.encode())

    def _assert_fails_closed(self, fake, status):
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_DEVICE_SELECTION)
        self.assertIn(f"CAPTURE_FAILED: {status}", out)
        self.assertEqual(fake.names(), ["IMV_EnumDevices"])
        self.assertNotIn("hardware", out)
        self.assertNotIn(FULL_SERIAL, out)

    def test_zero_devices_fails_closed(self):
        self._assert_fails_closed(FakeSdk(devices=[]), "NO_DEVICE")

    def test_two_matching_devices_fail_closed(self):
        devices = [_device(), _device(serial=b"DA0000001111")]
        self._assert_fails_closed(FakeSdk(devices=devices), "MULTIPLE_DEVICES")

    def test_expected_plus_other_device_fails_closed(self):
        devices = [_device(model=b"MV-OTHER", serial=b"XX0000001111"), _device()]
        self._assert_fails_closed(FakeSdk(devices=devices), "MULTIPLE_DEVICES")

    def test_unexpected_model_fails_closed(self):
        self._assert_fails_closed(FakeSdk(devices=[_device(model=b"MV-OTHER")]), "UNEXPECTED_DEVICE")

    def test_unexpected_transport_fails_closed(self):
        self._assert_fails_closed(FakeSdk(devices=[_device(camera_type=0, interface_type=1)]),
                                  "UNEXPECTED_TRANSPORT")

    def test_enumeration_error_preserves_code(self):
        fake = FakeSdk(enum_code=-101)
        result = ff.capture_one_frame(fake.ns)
        self.assertEqual(result.status, "ENUMERATION_FAILED")
        self.assertEqual(result.sdk_code, -101)
        self.assertEqual(result.sdk_code_name, "IMV_ERROR")
        self.assertEqual(fake.names(), ["IMV_EnumDevices"])
        self.assertEqual(result.cleanup, [])

    def test_serial_masked_in_output(self):
        _, out, _, _ = _run(FakeSdk())
        self.assertNotIn(FULL_SERIAL, out)
        self.assertIn("****7890", out)


class LifecycleTests(NoPhysicalSdkTestCase):
    def test_success_call_order(self):
        fake = FakeSdk()
        real_string_at = ctypes.string_at

        def recording_string_at(ptr, size):
            fake.calls.append(("copy", size))
            return real_string_at(ptr, size)

        with mock.patch.object(ff.ctypes, "string_at", side_effect=recording_string_at):
            result = ff.capture_one_frame(fake.ns)
        self.assertTrue(result.ok)
        self.assertEqual(fake.names(), [
            "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_GetEnumFeatureSymbol",
            "IMV_StartGrabbing", "IMV_GetFrame", "copy", "IMV_ReleaseFrame", "IMV_StopGrabbing",
            "IMV_Close", "IMV_DestroyHandle",
        ])
        self.assertEqual([step.name for step in result.cleanup], [
            "IMV_ReleaseFrame", "IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle",
        ])

    def test_exactly_one_get_frame_with_bounded_timeout(self):
        fake = FakeSdk()
        ff.capture_one_frame(fake.ns)
        get_frames = [call for call in fake.calls if call[0] == "IMV_GetFrame"]
        self.assertEqual(len(get_frames), 1)
        timeout = get_frames[0][2]
        self.assertEqual(timeout, ff.GET_FRAME_TIMEOUT_MS)
        self.assertTrue(2000 <= timeout <= 3000)

    def test_success_leaves_no_open_resources(self):
        fake = FakeSdk()
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_OK)
        self.assertEqual(fake.handles_alive, 0)
        self.assertFalse(fake.is_open)
        self.assertFalse(fake.is_grabbing)
        self.assertEqual(fake.frames_outstanding, 0)
        self.assertIn("FRAME_RELEASED_AND_CAMERA_SHUT_DOWN", out)

    def test_release_exactly_once_after_success(self):
        fake = FakeSdk()
        ff.capture_one_frame(fake.ns)
        self.assertEqual(fake.names().count("IMV_ReleaseFrame"), 1)

    def test_only_trigger_mode_is_read_and_no_feature_is_written(self):
        fake = FakeSdk()
        ff.capture_one_frame(fake.ns)
        reads = [call[1] for call in fake.calls if call[0] == "IMV_GetEnumFeatureSymbol"]
        self.assertEqual(reads, ["TriggerMode"])
        allowed = {
            "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_GetEnumFeatureSymbol",
            "IMV_StartGrabbing", "IMV_GetFrame", "IMV_ReleaseFrame", "IMV_StopGrabbing",
            "IMV_Close", "IMV_DestroyHandle",
        }
        self.assertTrue(set(fake.names()) <= allowed)

    def test_create_handle_failure_cleans_nothing(self):
        fake = FakeSdk(create_code=-101)
        result = ff.capture_one_frame(fake.ns)
        self.assertEqual(result.status, "CREATE_HANDLE_FAILED")
        self.assertEqual(fake.names(), ["IMV_EnumDevices", "IMV_CreateHandle"])
        self.assertEqual(result.cleanup, [])

    def test_open_failure_destroys_handle_only(self):
        fake = FakeSdk(open_code=-102)
        result = ff.capture_one_frame(fake.ns)
        self.assertEqual(result.status, "OPEN_FAILED")
        self.assertEqual(result.sdk_code_name, "IMV_INVALID_HANDLE")
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")
        self.assertNotIn("IMV_Close", fake.names())
        self.assertEqual(fake.handles_alive, 0)

    def test_start_grabbing_failure_cleans_acquired_resources(self):
        fake = FakeSdk(start_code=-101)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_CAPTURE_FAILED)
        self.assertIn("CAPTURE_FAILED: START_GRABBING_FAILED", out)
        names = fake.names()
        self.assertEqual(names[-3:], ["IMV_StartGrabbing", "IMV_Close", "IMV_DestroyHandle"])
        for not_expected in ("IMV_GetFrame", "IMV_ReleaseFrame", "IMV_StopGrabbing"):
            self.assertNotIn(not_expected, names)
        self.assertEqual(fake.handles_alive, 0)
        self.assertFalse(fake.is_open)

    def test_trigger_mode_on_refuses_without_changing_it(self):
        fake = FakeSdk(trigger=b"On")
        result = ff.capture_one_frame(fake.ns)
        self.assertEqual(result.status, "TRIGGER_MODE_ON")
        self.assertEqual(result.trigger_mode, "On")
        self.assertNotIn("IMV_StartGrabbing", fake.names())
        self.assertNotIn("IMV_GetFrame", fake.names())
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])

    def test_trigger_mode_read_failure_is_reported_and_capture_proceeds(self):
        fake = FakeSdk(trigger_code=-111)
        result = ff.capture_one_frame(fake.ns)
        self.assertTrue(result.ok)
        self.assertIn("unknown (read failed with code -111", result.trigger_mode)


class AcquisitionFailureTests(NoPhysicalSdkTestCase):
    def test_failed_get_frame_never_yields_frame(self):
        fake = FakeSdk(get_frame_code=-101)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_CAPTURE_FAILED)
        self.assertIn("CAPTURE_FAILED: GET_FRAME_FAILED", out)
        self.assertIn("(error code -101 IMV_ERROR)", out)
        self.assertNotIn("hardware", out)
        self.assertNotIn("RAW_FRAME_CAPTURED", out)
        self.assertNotIn("IMV_ReleaseFrame", fake.names())

    def test_get_frame_timeout_cleans_up(self):
        fake = FakeSdk(get_frame_code=ff.IMV_TIMEOUT)
        result = ff.capture_one_frame(fake.ns)
        self.assertFalse(result.ok)
        self.assertIsNone(result.frame)
        self.assertEqual(result.status, "GET_FRAME_FAILED")
        self.assertEqual(result.sdk_code_name, "IMV_TIMEOUT")
        self.assertEqual([s.name for s in result.cleanup], ["IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.handles_alive, 0)
        self.assertFalse(fake.is_grabbing)

    def test_native_error_is_not_retried(self):
        for kwargs, name in (({"get_frame_code": -101}, "IMV_GetFrame"),
                             ({"create_code": -101}, "IMV_CreateHandle"),
                             ({"open_code": -101}, "IMV_Open"),
                             ({"start_code": -101}, "IMV_StartGrabbing"),
                             ({"enum_code": -101}, "IMV_EnumDevices")):
            with self.subTest(name=name):
                fake = FakeSdk(**kwargs)
                ff.capture_one_frame(fake.ns)
                self.assertEqual(fake.names().count(name), 1)
                self.assertLessEqual(fake.names().count("IMV_GetFrame"), 1)

    def test_unexpected_exception_from_get_frame_cleans_up(self):
        fake = FakeSdk(raises={"IMV_GetFrame": OSError("access violation")})
        result = ff.capture_one_frame(fake.ns)
        self.assertEqual(result.status, ff.STATUS_UNEXPECTED_ERROR)
        self.assertIsNone(result.frame)
        self.assertNotIn("IMV_ReleaseFrame", fake.names())
        self.assertEqual(fake.handles_alive, 0)

    def test_keyboard_interrupt_during_get_frame_is_not_success(self):
        fake = FakeSdk(raises={"IMV_GetFrame": KeyboardInterrupt()})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_INTERRUPTED)
        self.assertIn("CAPTURE_FAILED: INTERRUPTED", out)
        self.assertNotIn("hardware", out)
        self.assertEqual(fake.names()[-3:], ["IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"])

    def test_keyboard_interrupt_during_cleanup_continues_cleanup(self):
        fake = FakeSdk(raises={"IMV_StopGrabbing": KeyboardInterrupt()})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_INTERRUPTED)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertIn("Cleanup IMV_StopGrabbing: FAILED (interrupted", out)


class FrameValidationTests(NoPhysicalSdkTestCase):
    def _assert_rejected_before_copy(self, fake, fragment):
        with mock.patch.object(ff.ctypes, "string_at") as string_at:
            result = ff.capture_one_frame(fake.ns)
        string_at.assert_not_called()
        self.assertEqual(result.status, "INVALID_FRAME")
        self.assertIn(fragment, result.message)
        self.assertIsNone(result.frame)
        self.assertEqual(fake.names().count("IMV_ReleaseFrame"), 1)
        self.assertEqual(fake.frames_outstanding, 0)
        self.assertEqual(fake.handles_alive, 0)

    def test_null_pointer_rejected(self):
        self._assert_rejected_before_copy(FakeSdk(null_data=True), "NULL")

    def test_nonzero_frame_status_rejected(self):
        self._assert_rejected_before_copy(FakeSdk(status=1), "frame status 1")

    def test_invalid_dimensions_rejected(self):
        for width, height in ((0, 2), (4, 0), (ff.MAX_DIMENSION + 1, 2), (4, ff.MAX_DIMENSION + 1)):
            with self.subTest(width=width, height=height):
                self._assert_rejected_before_copy(FakeSdk(width=width, height=height), "dimensions")

    def test_oversized_payload_rejected(self):
        self._assert_rejected_before_copy(FakeSdk(size=ff.MAX_PAYLOAD_BYTES + 1), "exceeds")

    def test_zero_payload_rejected(self):
        self._assert_rejected_before_copy(FakeSdk(size=0), "invalid payload size 0")

    def test_payload_smaller_than_layout_rejected(self):
        self._assert_rejected_before_copy(FakeSdk(size=7), "smaller than 8 bytes")
        self._assert_rejected_before_copy(
            FakeSdk(pixel_format=MONO10, payload=bytes(16), size=15), "smaller than 16 bytes")
        self._assert_rejected_before_copy(
            FakeSdk(pixel_format=MONO12_PACKED, payload=bytes(12), size=11), "smaller than 12 bytes")

    def test_failed_release_is_visible(self):
        fake = FakeSdk(release_code=-104)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_CLEANUP_FAILED)
        self.assertEqual(fake.names().count("IMV_ReleaseFrame"), 1)
        self.assertIn("Cleanup IMV_ReleaseFrame: FAILED with code -104", out)
        self.assertIn("CLEANUP_FAILURE", out)
        self.assertNotIn("FRAME_RELEASED_AND_CAMERA_SHUT_DOWN", out)
        self.assertEqual(fake.names()[-3:], ["IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"])

    def test_shutdown_failures_are_visible_and_later_steps_still_run(self):
        for name, kwargs in (("IMV_StopGrabbing", {"stop_code": -101}),
                             ("IMV_Close", {"close_code": -101}),
                             ("IMV_DestroyHandle", {"destroy_code": -101})):
            with self.subTest(name=name):
                fake = FakeSdk(**kwargs)
                code, out, _, _ = _run(fake)
                self.assertEqual(code, ff.EXIT_CLEANUP_FAILED)
                self.assertIn(f"Cleanup {name}: FAILED with code -101", out)
                self.assertIn("CLEANUP_FAILURE", out)
                self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")

    def test_shutdown_exception_is_visible(self):
        fake = FakeSdk(raises={"IMV_Close": OSError("boom")})
        result = ff.capture_one_frame(fake.ns)
        self.assertFalse(result.ok)
        close_step = [s for s in result.cleanup if s.name == "IMV_Close"][0]
        self.assertIn("OSError", close_step.error)
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")


class FrameContentTests(NoPhysicalSdkTestCase):
    def test_copied_bytes_survive_native_release(self):
        fake = FakeSdk()
        result = ff.capture_one_frame(fake.ns)
        self.assertIsInstance(result.frame.data, bytes)
        self.assertEqual(result.frame.data, MONO8_PAYLOAD)
        self.assertEqual(bytes(fake.native_buffer), b"\xee" * len(MONO8_PAYLOAD))

    def test_sha256_matches_copied_bytes(self):
        fake = FakeSdk()
        code, out, _, _ = _run(fake)
        digest = hashlib.sha256(MONO8_PAYLOAD).hexdigest()
        self.assertEqual(code, ff.EXIT_OK)
        self.assertIn(f"SHA-256:            {digest}", out)

    def test_metadata_reported(self):
        fake = FakeSdk()
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_OK)
        for fragment in ("Model:              MV-A7200MU130", "Frame ID (blockId): 42",
                         "Width x Height:     4 x 2", "Pixel format:       0x01080001 (Mono8)",
                         "Payload bytes:      8", "Acquisition source: hardware",
                         "TriggerMode (read): Off", "CAPTURE_STATUS: RAW_FRAME_CAPTURED"):
            self.assertIn(fragment, out)
        self.assertIn("No claim is made", out)

    def test_verified_mono8_statistics(self):
        fake = FakeSdk()
        result = ff.capture_one_frame(fake.ns)
        frame = result.frame
        self.assertEqual(frame.decode_status, ff.DECODE_MONO8)
        self.assertEqual((frame.pixel_min, frame.pixel_max), (0, 255))
        self.assertAlmostEqual(frame.pixel_mean, sum(MONO8_PAYLOAD) / 8)

    def test_mono8_with_padding_not_decoded(self):
        fake = FakeSdk(padding=(2, 0))
        frame = ff.capture_one_frame(fake.ns).frame
        self.assertEqual(frame.decode_status, ff.DECODE_UNAVAILABLE)
        self.assertIsNone(frame.pixel_mean)

    def test_mono8_with_extra_payload_not_decoded(self):
        fake = FakeSdk(payload=MONO8_PAYLOAD + b"\x01\x02")
        result = ff.capture_one_frame(fake.ns)
        self.assertTrue(result.ok)
        self.assertEqual(result.frame.payload_bytes, 10)
        self.assertEqual(result.frame.decode_status, ff.DECODE_UNAVAILABLE)

    def test_unknown_pixel_format_not_interpreted_as_mono8(self):
        fake = FakeSdk(pixel_format=UNKNOWN_FORMAT)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, ff.EXIT_OK)
        self.assertIn("(unrecognized)", out)
        self.assertIn("PIXEL_DECODE_UNAVAILABLE", out)
        self.assertNotIn("min=", out)

    def test_mono10_raw_captured_without_decode(self):
        payload = bytes(range(16))
        fake = FakeSdk(pixel_format=MONO10, payload=payload)
        result = ff.capture_one_frame(fake.ns)
        self.assertTrue(result.ok)
        self.assertEqual(result.frame.pixel_format_label, "Mono10")
        self.assertEqual(result.frame.decode_status, ff.DECODE_UNAVAILABLE)
        self.assertEqual(result.frame.data, payload)
        self.assertIsNone(result.frame.pixel_min)


class SourceSafetyTests(unittest.TestCase):
    def test_source_has_no_configuration_writes_or_extra_activity(self):
        source = (HW3_DIR / "irayple_first_frame.py").read_text(encoding="utf-8")
        for forbidden in ("SetIntFeature", "SetDoubleFeature", "SetBoolFeature", "SetEnumFeature",
                          "SetStringFeature", "ExecuteCommand", "StartGrabbingEx", "AttachGrabbing",
                          "interfaceTypeAll", "modeByIndex", "threading", "subprocess", "environ",
                          "open(", "SaveImage"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
