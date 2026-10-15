import builtins
import contextlib
import ctypes
import importlib
import io
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_exposure_sweep as es  # noqa: E402

FULL_SERIAL = "DA1234567890"
FULL_CAMERA_KEY = "Huaray Technology:" + FULL_SERIAL
MONO8 = 0x01080001
MONO10 = 0x01100003
NAN = float("nan")
# 4 x 2 Mono8 frames. After release the fake SDK overwrites its buffer with 0xEE (238 DN),
# which would change every statistic below if pixel memory were read after release.
PAYLOAD_1 = bytes([0, 10, 20, 30, 40, 50, 60, 70])          # mean 35.0, no saturation
PAYLOAD_2 = bytes([0, 50, 100, 150, 200, 250, 255, 255])    # mean 157.5, 2/8 saturated

SDK_CALLS = (
    "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_FeatureIsAvailable", "IMV_FeatureIsReadable",
    "IMV_FeatureIsWriteable", "IMV_GetEnumFeatureSymbol", "IMV_GetDoubleFeatureValue",
    "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax", "IMV_SetDoubleFeatureValue",
    "IMV_StartGrabbing", "IMV_GetFrame", "IMV_ReleaseFrame", "IMV_StopGrabbing", "IMV_Close",
    "IMV_DestroyHandle",
)
FORBIDDEN_CALLS = (
    "IMV_SetIntFeatureValue", "IMV_SetBoolFeatureValue", "IMV_SetEnumFeatureValue",
    "IMV_SetEnumFeatureSymbol", "IMV_SetStringFeatureValue", "IMV_ExecuteCommandFeature",
    "IMV_StartGrabbingEx", "IMV_AttachGrabbing", "IMV_OpenEx", "IMV_CloneFrame",
)
BOOL_CALLS = ("IMV_FeatureIsAvailable", "IMV_FeatureIsReadable", "IMV_FeatureIsWriteable")
WRITE = "IMV_SetDoubleFeatureValue"
READ_VALUE = "IMV_GetDoubleFeatureValue"


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
    """Fake vendor SDK with camera state. Every call goes into one ordered log.

    codes[name] may be an int (every call) or a list (one code per successive call, then 0).
    raises[name] may be an exception (every call) or a list (per successive call, None = no raise).
    apply_set(requested, write_index) returns the value the fake camera actually stores.
    """

    def __init__(self, devices=None, codes=None, raises=None, enums=None, available=None, readable=None,
                 writeable=True, exposure=(5000.0, 16.0, 1000000.0), apply_set=None,
                 payloads=(PAYLOAD_1, PAYLOAD_2), width=4, height=2, pixel_formats=(MONO8, MONO8),
                 padding=(0, 0), status=0):
        self.devices = [_device()] if devices is None else devices
        self.codes = codes or {}
        self.raises = raises or {}
        self.enums = {"TriggerMode": b"Off", "ExposureAuto": b"Off", "GainAuto": b"Off"}
        self.enums.update(enums or {})
        self.available = set(self.enums) if available is None else set(available)
        self.readable = set(self.available) if readable is None else set(readable)
        self.writeable = writeable
        self.current, self.minimum, self.maximum = exposure
        self.apply_set = apply_set or (lambda requested, index: requested)
        self.payloads = payloads
        self.width, self.height = width, height
        self.pixel_formats = pixel_formats
        self.padding = padding
        self.status = status
        self.counts = {}
        self.calls = []
        self.handles_alive = 0
        self.is_open = False
        self.is_grabbing = False
        self.frames_outstanding = {}
        self.released_block_ids = []
        self.exposure_at_frame = []
        self.violations = []
        self.ns = self._namespace()

    def names(self):
        return [call[0] for call in self.calls]

    def writes(self):
        return [call[2] for call in self.calls if call[0] == WRITE]

    def _next(self, table, name, index, default):
        entry = table.get(name, default)
        if isinstance(entry, list):
            return entry[index] if index < len(entry) else default
        return entry

    def _call(self, name, *args):
        index = self.counts.get(name, 0)
        self.counts[name] = index + 1
        self.calls.append((name,) + args)
        exc = self._next(self.raises, name, index, None)
        if exc is not None:
            raise exc
        if name in FORBIDDEN_CALLS:
            self.violations.append(name)
        code = self._next(self.codes, name, index, 0)
        if code == 0:
            result = getattr(self, "_on_" + name, lambda *a: None)(*args)
            if name in BOOL_CALLS:
                return result
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

    def _on_IMV_FeatureIsAvailable(self, name):
        return name in self.available

    def _on_IMV_FeatureIsReadable(self, name):
        return name in self.readable

    def _on_IMV_FeatureIsWriteable(self, name):
        return name == "ExposureTime" and self.writeable

    def _on_IMV_GetEnumFeatureSymbol(self, name, symbol):
        assert isinstance(symbol, FakeString), type(symbol)
        symbol.str = self.enums[name]

    def _write_double(self, name, out, value):
        assert name == "ExposureTime", name
        assert isinstance(out, ctypes.c_double), type(out)
        out.value = value

    def _on_IMV_GetDoubleFeatureValue(self, name, out):
        self._write_double(name, out, self.current)

    def _on_IMV_GetDoubleFeatureMin(self, name, out):
        self._write_double(name, out, self.minimum)

    def _on_IMV_GetDoubleFeatureMax(self, name, out):
        self._write_double(name, out, self.maximum)

    def _on_IMV_SetDoubleFeatureValue(self, name, value):
        # Vendor wrapper signature: (str pFeatureName, float doubleValue); it wraps c_double itself.
        if name != "ExposureTime" or not isinstance(value, float) or self.is_grabbing or not self.is_open:
            self.violations.append(("bad write", name, value, self.is_grabbing))
        self.current = self.apply_set(value, self.counts[WRITE] - 1)

    def _on_IMV_StartGrabbing(self):
        if self.is_grabbing:
            self.violations.append("double StartGrabbing")
        self.is_grabbing = True

    def _on_IMV_StopGrabbing(self):
        self.is_grabbing = False

    def _on_IMV_GetFrame(self, frame, timeout_ms):
        if not self.is_grabbing:
            self.violations.append("GetFrame while not grabbing")
        index = self.counts["IMV_GetFrame"] - 1
        payload = self.payloads[index]
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        frame.pData = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        info = frame.frameInfo
        info.blockId = 101 + index
        info.status = self.status
        info.width, info.height = self.width, self.height
        info.size = len(payload)
        info.pixelFormat = self.pixel_formats[index]
        info.paddingX, info.paddingY = self.padding
        self.frames_outstanding[101 + index] = buffer
        self.exposure_at_frame.append(self.current)

    def _on_IMV_ReleaseFrame(self, frame):
        block_id = int(frame.frameInfo.blockId)
        buffer = self.frames_outstanding.pop(block_id, None)
        if buffer is None:
            self.violations.append(("release of a frame not outstanding", block_id))
            return
        self.released_block_ids.append(block_id)
        # The SDK may reuse its buffer after release; overwrite it to prove nothing still reads it.
        ctypes.memset(buffer, 0xEE, len(buffer))

    def _namespace(self):
        fake = self

        def method(name):
            return lambda self, *args: fake._call(name, *args)

        attrs = {name: method(name) for name in SDK_CALLS + FORBIDDEN_CALLS}
        attrs["IMV_EnumDevices"] = staticmethod(lambda device_list, interface_type: fake._call(
            "IMV_EnumDevices", device_list, interface_type))
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
            IMV_EPixelType=SimpleNamespace(
                gvspPixelTypeUndefined=-1, gvspPixelMono8=MONO8, gvspPixelMono10=MONO10,
            ),
            IMV_Frame=FakeFrame,
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


def _sweep(fake):
    """Run the sweep with ctypes.string_at logged into the fake's call order."""
    real_string_at = ctypes.string_at

    def logged_string_at(address, size):
        fake.calls.append(("string_at", size))
        return real_string_at(address, size)

    with mock.patch.object(es.ctypes, "string_at", side_effect=logged_string_at):
        return es.run_exposure_sweep(fake.ns)


def _run(fake=None, argv=("--sweep-once", "--unit-us-verified"), response=es.CONFIRMATION_PHRASE,
         interactive=True, loader=None, input_fn=None):
    if loader is None:
        loader = mock.Mock(return_value=fake.ns if fake else None)
    if input_fn is None:
        input_fn = mock.Mock(return_value=response)
    stdin = SimpleNamespace(isatty=lambda: interactive)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = es.main(list(argv), sdk_loader=loader, input_fn=input_fn, stdin=stdin)
    return code, out.getvalue(), loader, input_fn


class NoPhysicalSdkTestCase(unittest.TestCase):
    """Any attempt to load the real DLL or vendor wrapper during a test fails loudly."""

    def setUp(self):
        guard = AssertionError("real SDK access attempted in an offline test")
        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch.object(es.discovery, "load_vendor_sdk", side_effect=guard),
        ):
            target.start()
            self.addCleanup(target.stop)

    def tearDown(self):
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)

    def assert_clean_shutdown(self, fake, result):
        self.assertEqual(fake.violations, [])
        self.assertEqual(fake.frames_outstanding, {})
        self.assertFalse(fake.is_grabbing)
        self.assertFalse(fake.is_open)
        self.assertEqual(fake.handles_alive, 0)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])

    def assert_restored(self, fake, result):
        self.assertEqual(result.restore.status, es.RESTORE_VERIFIED)
        self.assertEqual(fake.writes()[-1], 5000.0)
        self.assertEqual(fake.current, 5000.0)
        restore_index = len(fake.names()) - 1 - fake.names()[::-1].index(WRITE)
        self.assertEqual(fake.names()[restore_index + 1], READ_VALUE)
        self.assertLess(restore_index, fake.names().index("IMV_Close"))


# 1. No SDK load or hardware access on import.
class ImportSafetyTests(unittest.TestCase):
    def test_import_performs_no_sdk_operations(self):
        with mock.patch.object(builtins, "input") as input_fn, \
                mock.patch("ctypes.WinDLL") as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir, \
                mock.patch("importlib.import_module") as import_module:
            importlib.reload(es)
        for patched in (win_dll, cdll, add_dir, chdir, import_module, input_fn):
            patched.assert_not_called()
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


# 2-4. Manual authorization gate.
class AuthorizationGateTests(NoPhysicalSdkTestCase):
    def test_missing_flags_refuse_without_hardware(self):
        for argv in ((), ("--sweep-once",), ("--unit-us-verified",)):
            with self.subTest(argv=argv):
                fake = FakeSdk()
                code, out, loader, input_fn = _run(fake, argv=argv)
                self.assertEqual(code, es.EXIT_REFUSED)
                self.assertIn("No camera was accessed", out)
                loader.assert_not_called()
                input_fn.assert_not_called()
                self.assertEqual(fake.calls, [])

    def test_default_loader_not_used_without_flags(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main([], input_fn=mock.Mock(), stdin=SimpleNamespace(isatty=lambda: True))
        self.assertEqual(code, es.EXIT_REFUSED)
        es.discovery.load_vendor_sdk.assert_not_called()

    def test_noninteractive_refuses(self):
        fake = FakeSdk()
        with mock.patch.dict(os.environ, {"LUMEN_APPROVE": "1", "CI": "true"}):
            code, out, loader, input_fn = _run(fake, interactive=False)
        self.assertEqual(code, es.EXIT_REFUSED)
        self.assertIn("interactive terminal is required", out)
        loader.assert_not_called()
        input_fn.assert_not_called()
        self.assertEqual(fake.calls, [])

    def test_stdin_without_isatty_refuses(self):
        fake = FakeSdk()
        loader = mock.Mock(return_value=fake.ns)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main(["--sweep-once", "--unit-us-verified"], sdk_loader=loader,
                           input_fn=mock.Mock(return_value=es.CONFIRMATION_PHRASE), stdin=object())
        self.assertEqual(code, es.EXIT_REFUSED)
        loader.assert_not_called()

    def test_exact_confirmation_required(self):
        for response in ("", "run exposure sweep", "RUN EXPOSURE SWEEP ", " RUN EXPOSURE SWEEP", "yes",
                         "CAPTURE ONE FRAME"):
            with self.subTest(response=response):
                fake = FakeSdk()
                code, out, loader, _ = _run(fake, response=response)
                self.assertEqual(code, es.EXIT_REFUSED)
                self.assertIn("confirmation phrase did not match", out)
                loader.assert_not_called()
                self.assertEqual(fake.calls, [])

    def test_eof_or_interrupt_at_prompt_refuses(self):
        for exc in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                fake = FakeSdk()
                code, out, loader, _ = _run(fake, input_fn=mock.Mock(side_effect=exc))
                self.assertEqual(code, es.EXIT_REFUSED)
                loader.assert_not_called()

    def test_plan_and_restore_shown_before_prompt(self):
        captured = {}
        out = io.StringIO()

        def answer(prompt):
            captured["before_prompt"] = out.getvalue()
            captured["prompt"] = prompt
            return "no"

        with contextlib.redirect_stdout(out):
            es.main(["--sweep-once", "--unit-us-verified"], sdk_loader=mock.Mock(), input_fn=answer,
                    stdin=SimpleNamespace(isatty=lambda: True))
        shown = captured["before_prompt"]
        self.assertIn("set ExposureTime = 1000.0 us", shown)
        self.assertIn("set ExposureTime = 5000.0 us", shown)
        self.assertIn("Restore:", shown)
        self.assertIn("originally read ExposureTime", shown)
        self.assertIn("human-verified in MV Viewer", shown)
        self.assertIn(es.CONFIRMATION_PHRASE, captured["prompt"])

    def test_confirmed_run_loads_sdk_once(self):
        fake = FakeSdk()
        code, _, loader, input_fn = _run(fake)
        self.assertEqual(code, es.EXIT_OK)
        loader.assert_called_once_with()
        input_fn.assert_called_once()

    def test_sdk_load_failure_is_controlled(self):
        loader = mock.Mock(side_effect=es.discovery.DiscoveryError("HuaRay SDK file not found: X"))
        code, out, _, _ = _run(loader=loader)
        self.assertEqual(code, es.EXIT_SWEEP_FAILED)
        self.assertIn("SDK_LOAD_FAILED", out)
        self.assertIn("SWEEP_STATUS: FAILED", out)


# 5. Device selection.
class DeviceSelectionTests(NoPhysicalSdkTestCase):
    def test_wrong_device_refuses_before_open(self):
        cases = (
            ([], "NO_DEVICE"),
            ([_device(), _device(serial=b"DA0000001111")], "MULTIPLE_DEVICES"),
            ([_device(model=b"MV-OTHER")], "UNEXPECTED_DEVICE"),
            ([_device(camera_type=0, interface_type=1)], "UNEXPECTED_TRANSPORT"),
        )
        for devices, status in cases:
            with self.subTest(status=status):
                fake = FakeSdk(devices=devices)
                code, out, _, _ = _run(fake)
                self.assertEqual(code, es.EXIT_DEVICE_SELECTION)
                self.assertIn(f"SWEEP_FAILURE: {status}", out)
                self.assertEqual(fake.names(), ["IMV_EnumDevices"])
                self.assertEqual(fake.writes(), [])

    def test_usb3_only_and_camera_key(self):
        fake = FakeSdk()
        result = _sweep(fake)
        self.assertTrue(result.ok)
        enums = [call for call in fake.calls if call[0] == "IMV_EnumDevices"]
        self.assertEqual(len(enums), 1)
        self.assertEqual(enums[0][2], 0x2)
        create_call = fake.calls[1]
        self.assertEqual(create_call[1], 1)  # modeByCameraKey
        self.assertEqual(create_call[2].value, FULL_CAMERA_KEY.encode())

    def test_identifiers_not_printed_in_full(self):
        _, out, _, _ = _run(FakeSdk())
        self.assertNotIn(FULL_SERIAL, out)
        self.assertIn("****7890", out)


# 6, 11-13, 15-18, 25. Successful protocol.
class SuccessfulSweepTests(NoPhysicalSdkTestCase):
    def test_exact_call_sequence(self):
        fake = FakeSdk()
        result = _sweep(fake)
        self.assertTrue(result.ok)
        enum_reads = ["IMV_FeatureIsAvailable", "IMV_FeatureIsReadable", "IMV_GetEnumFeatureSymbol"] * 3
        per_frame = [WRITE, READ_VALUE, "IMV_StartGrabbing", "IMV_GetFrame", "string_at",
                     "IMV_ReleaseFrame", "IMV_StopGrabbing"]
        self.assertEqual(fake.names(), [
            "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", *enum_reads,
            READ_VALUE, "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax", "IMV_FeatureIsWriteable",
            *per_frame, *per_frame,
            WRITE, READ_VALUE,
            "IMV_Close", "IMV_DestroyHandle",
        ])
        self.assert_clean_shutdown(fake, result)

    def test_one_session_two_frames(self):
        fake = FakeSdk()
        _sweep(fake)
        for name, count in (("IMV_EnumDevices", 1), ("IMV_CreateHandle", 1), ("IMV_Open", 1),
                            ("IMV_GetFrame", 2), ("IMV_ReleaseFrame", 2), ("IMV_StartGrabbing", 2),
                            ("IMV_StopGrabbing", 2), ("IMV_Close", 1), ("IMV_DestroyHandle", 1)):
            with self.subTest(name=name):
                self.assertEqual(fake.names().count(name), count)

    def test_preconditions_read_in_order(self):
        fake = FakeSdk()
        _sweep(fake)
        reads = [call[1] for call in fake.calls if call[0] == "IMV_GetEnumFeatureSymbol"]
        self.assertEqual(reads, ["TriggerMode", "ExposureAuto", "GainAuto"])

    def test_limits_checked_before_any_write(self):
        fake = FakeSdk()
        _sweep(fake)
        names = fake.names()
        first_write = names.index(WRITE)
        for name in ("IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax", "IMV_FeatureIsWriteable"):
            self.assertLess(names.index(name), first_write)

    def test_two_writes_in_order_then_restore(self):
        fake = FakeSdk()
        _sweep(fake)
        self.assertEqual(fake.writes(), [1000.0, 5000.0, 5000.0])
        for call in fake.calls:
            if call[0] == WRITE:
                self.assertEqual(call[1], "ExposureTime")
                self.assertIs(type(call[2]), float)

    def test_readback_after_each_write(self):
        fake = FakeSdk()
        result = _sweep(fake)
        names = fake.names()
        for index, name in enumerate(names):
            if name == WRITE:
                self.assertEqual(names[index + 1], READ_VALUE)
        self.assertEqual(result.applied, [es.AppliedExposure(1000.0, 0, 1000.0),
                                          es.AppliedExposure(5000.0, 0, 5000.0)])

    def test_frames_acquired_at_applied_exposure(self):
        fake = FakeSdk()
        _sweep(fake)
        self.assertEqual(fake.exposure_at_frame, [1000.0, 5000.0])

    def test_buffers_copied_before_release_and_released_once(self):
        fake = FakeSdk()
        _sweep(fake)
        names = fake.names()
        frame_events = [name for name in names if name in ("IMV_GetFrame", "string_at", "IMV_ReleaseFrame")]
        self.assertEqual(frame_events, ["IMV_GetFrame", "string_at", "IMV_ReleaseFrame"] * 2)
        self.assertEqual(fake.released_block_ids, [101, 102])
        copies = [call[1] for call in fake.calls if call[0] == "string_at"]
        self.assertEqual(copies, [8, 8])

    def test_statistics_come_from_pre_release_data(self):
        fake = FakeSdk()
        result = _sweep(fake)
        first, second = result.frames
        self.assertEqual((first.min_dn, first.max_dn, first.mean_dn, first.saturation_fraction), (0, 70, 35.0, 0.0))
        self.assertEqual((second.min_dn, second.max_dn, second.mean_dn, second.saturation_fraction),
                         (0, 255, 157.5, 0.25))
        for frame in result.frames:
            self.assertNotEqual(frame.max_dn, 0xEE)  # released buffers were overwritten with 0xEE

    def test_frame_metadata_reflects_sdk(self):
        fake = FakeSdk()
        result = _sweep(fake)
        self.assertEqual([frame.frame_id for frame in result.frames], [101, 102])
        self.assertEqual([frame.requested_exposure for frame in result.frames], [1000.0, 5000.0])
        self.assertEqual([frame.readback_exposure for frame in result.frames], [1000.0, 5000.0])
        for frame in result.frames:
            self.assertEqual((frame.width, frame.height, frame.pixel_format), (4, 2, MONO8))
            self.assertEqual(frame.pixel_format_label, "Mono8")
            self.assertEqual(frame.acquisition_source, "hardware")

    def test_report_contents(self):
        code, out, _, _ = _run(FakeSdk())
        self.assertEqual(code, es.EXIT_OK)
        for expected in (
            "Model:               MV-A7200MU130",
            "Original ExposureTime: 5000.0 (SDK min 16.0, max 1000000.0)",
            "human-verified in MV Viewer",
            "Set ExposureTime 1000.0: IMV_OK; read back 1000.0",
            "Set ExposureTime 5000.0: IMV_OK; read back 5000.0",
            "Frame 1: requested 1000.0, read back 1000.0",
            "Frame ID (blockId): 101",
            "Frame ID (blockId): 102",
            "4 x 2 0x01080001 (Mono8), padding 0/0, 8 bytes",
            "Source:             hardware",
            "min DN 0, max DN 70, mean DN 35.0000, saturation fraction 0.000000",
            "min DN 0, max DN 255, mean DN 157.5000, saturation fraction 0.250000",
            "Observed mean DN change (frame 2 - frame 1): +122.5000 DN",
            "Observed mean DN ratio (frame 2 / frame 1): 4.5000",
            "linearity not assumed",
            "Exposure restoration: RESTORED_AND_VERIFIED (original 5000.0, read back 5000.0)",
            "Lifecycle IMV_ReleaseFrame (frame 1): OK",
            "Lifecycle IMV_StopGrabbing (frame 2): OK",
            "Lifecycle IMV_Close: OK",
            "Lifecycle IMV_DestroyHandle: OK",
            "SWEEP_STATUS: SUCCESS",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, out)
        for forbidden in ("lux", "irradiance", "calibrated sensitivity", "read noise"):
            self.assertNotIn(forbidden, out.lower())

    def test_zero_mean_first_frame_ratio_undefined(self):
        fake = FakeSdk(payloads=(bytes(8), PAYLOAD_2))
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_OK)
        self.assertIn("ratio: undefined (frame 1 mean is 0 DN)", out)

    def test_original_within_tolerance_restored_exactly(self):
        original = 5000.0 + 1e-9
        fake = FakeSdk(exposure=(original, 16.0, 1000000.0))
        result = _sweep(fake)
        self.assertTrue(result.ok)
        self.assertEqual(fake.writes()[-1], original)

    def test_no_files_or_network(self):
        fake = FakeSdk()
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")), \
                mock.patch("socket.socket", side_effect=AssertionError("network")):
            result = _sweep(fake)
        self.assertTrue(result.ok)


# 7-11. Fail-closed preconditions: no write is ever issued.
class PreconditionTests(NoPhysicalSdkTestCase):
    def _assert_no_write(self, fake, status):
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_PRECONDITION)
        self.assertIn(f"SWEEP_FAILURE: {status}", out)
        self.assertEqual(fake.writes(), [])
        for name in ("IMV_StartGrabbing", "IMV_GetFrame"):
            self.assertNotIn(name, fake.names())
        self.assertIn("Exposure restoration: NOT_REQUIRED", out)
        self.assertIn("SWEEP_STATUS: FAILED", out)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.handles_alive, 0)

    def test_trigger_mode_on_prevents_writes(self):
        self._assert_no_write(FakeSdk(enums={"TriggerMode": b"On"}), "TRIGGER_MODE_NOT_OFF")

    def test_exposure_auto_not_off_prevents_writes(self):
        for value in (b"Continuous", b"Once"):
            with self.subTest(value=value):
                self._assert_no_write(FakeSdk(enums={"ExposureAuto": value}), "EXPOSURE_AUTO_NOT_OFF")

    def test_gain_auto_not_off_prevents_writes(self):
        self._assert_no_write(FakeSdk(enums={"GainAuto": b"Continuous"}), "GAIN_AUTO_NOT_OFF")

    def test_unavailable_or_ambiguous_features_prevent_writes(self):
        cases = (
            FakeSdk(available={"TriggerMode", "ExposureAuto"}),
            FakeSdk(readable={"TriggerMode", "GainAuto"}),
            FakeSdk(enums={"GainAuto": b""}),
            FakeSdk(codes={"IMV_GetEnumFeatureSymbol": [0, -111]}),
        )
        for index, fake in enumerate(cases):
            with self.subTest(case=index):
                self._assert_no_write(fake, "FEATURE_UNAVAILABLE")

    def test_original_exposure_not_5000_prevents_writes(self):
        for original in (4000.0, 5000.01, 4999.99, 1000.0):
            with self.subTest(original=original):
                self._assert_no_write(FakeSdk(exposure=(original, 16.0, 1000000.0)),
                                      "ORIGINAL_EXPOSURE_UNEXPECTED")

    def test_invalid_limits_prevent_writes(self):
        cases = (
            ((NAN, 16.0, 1000000.0), es.exposure_info.STATUS_INVALID_VALUE),
            ((5000.0, NAN, 1000000.0), es.exposure_info.STATUS_INVALID_VALUE),
            ((5000.0, 16.0, 16.0), es.exposure_info.STATUS_INVALID_RANGE),
            ((5000.0, 16.0, 4000.0), es.exposure_info.STATUS_OUT_OF_RANGE),
        )
        for exposure, status in cases:
            with self.subTest(exposure=exposure):
                self._assert_no_write(FakeSdk(exposure=exposure), status)

    def test_planned_value_outside_live_limits_prevents_writes(self):
        self._assert_no_write(FakeSdk(exposure=(5000.0, 1500.0, 1000000.0)), "PLANNED_EXPOSURE_OUT_OF_RANGE")

    def test_not_writeable_prevents_writes(self):
        self._assert_no_write(FakeSdk(writeable=False), "EXPOSURE_NOT_WRITEABLE")

    def test_limit_read_failure_prevents_writes(self):
        fake = FakeSdk(codes={"IMV_GetDoubleFeatureMax": -111})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_SWEEP_FAILED)
        self.assertIn("EXPOSURE_QUERY_FAILED", out)
        self.assertIn("(error code -111 IMV_INVALID_ACCESS)", out)
        self.assertEqual(fake.writes(), [])


# 14, 19-21. Failures after writes begin always restore.
class RestoreOnFailureTests(NoPhysicalSdkTestCase):
    def test_readback_mismatch_is_reported_not_success(self):
        fake = FakeSdk(apply_set=lambda value, index: 998.4 if index == 0 else value)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_SWEEP_FAILED)
        self.assertIn("SWEEP_FAILURE: EXPOSURE_READBACK_MISMATCH", out)
        self.assertIn("requested ExposureTime 1000.0, SDK read back 998.4", out)
        self.assertIn("Set ExposureTime 1000.0: IMV_OK; read back 998.4", out)
        self.assertNotIn("SWEEP_STATUS: SUCCESS", out)
        self.assertNotIn("IMV_GetFrame", fake.names())
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertEqual(fake.current, 5000.0)

    def test_second_readback_mismatch_reported(self):
        fake = FakeSdk(apply_set=lambda value, index: 5000.5 if index == 1 else value)
        result = _sweep(fake)
        self.assertEqual(result.status, "EXPOSURE_READBACK_MISMATCH")
        self.assertFalse(result.ok)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(fake.names().count("IMV_GetFrame"), 1)
        self.assert_restored(fake, result)

    def test_failure_during_first_frame_restores(self):
        fake = FakeSdk(codes={"IMV_GetFrame": [-119]})
        result = _sweep(fake)
        self.assertEqual(result.status, "GET_FRAME_FAILED")
        self.assertEqual(result.sdk_code_name, "IMV_TIMEOUT")
        self.assertEqual(result.frames, [])
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertNotIn("IMV_ReleaseFrame", fake.names())
        self.assert_restored(fake, result)
        self.assert_clean_shutdown(fake, result)

    def test_failure_during_second_frame_restores(self):
        fake = FakeSdk(codes={"IMV_GetFrame": [0, -119]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_SWEEP_FAILED)
        self.assertIn("SWEEP_FAILURE: GET_FRAME_FAILED", out)
        self.assertIn("Frame 1: requested 1000.0", out)
        self.assertEqual(fake.writes(), [1000.0, 5000.0, 5000.0])
        self.assertEqual(fake.released_block_ids, [101])
        self.assertIn("Exposure restoration: RESTORED_AND_VERIFIED", out)
        self.assertIn("SWEEP_STATUS: FAILED", out)

    def test_invalid_frame_released_in_cleanup_without_copy(self):
        for kwargs in ({"pixel_formats": (MONO8, MONO10)}, {"padding": (2, 0)}, {"status": 1}):
            with self.subTest(kwargs=kwargs):
                fake = FakeSdk(**kwargs)
                result = _sweep(fake)
                self.assertFalse(result.ok)
                self.assertIn(result.status, ("NOT_MONO8", "INVALID_LAYOUT", "INVALID_FRAME"))
                acquired = fake.names().count("IMV_GetFrame")
                self.assertEqual(fake.names().count("string_at"), acquired - 1)
                self.assertEqual(fake.names().count("IMV_ReleaseFrame"), acquired)
                self.assert_restored(fake, result)
                self.assert_clean_shutdown(fake, result)

    def test_write_failure_reported_and_cleanup_attempted(self):
        fake = FakeSdk(codes={WRITE: [-111]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_SWEEP_FAILED)
        self.assertIn("SWEEP_FAILURE: EXPOSURE_WRITE_FAILED", out)
        self.assertIn("(error code -111 IMV_INVALID_ACCESS)", out)
        self.assertIn("Set ExposureTime 1000.0: FAILED with code -111; read back not read", out)
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertNotIn("IMV_StartGrabbing", fake.names())
        self.assertIn("Exposure restoration: RESTORED_AND_VERIFIED", out)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])

    def test_write_exception_still_restores(self):
        fake = FakeSdk(raises={WRITE: [OSError("access violation")]})
        result = _sweep(fake)
        self.assertEqual(result.status, es.STATUS_UNEXPECTED_ERROR)
        self.assertIn("access violation", result.message)
        self.assert_restored(fake, result)
        self.assert_clean_shutdown(fake, result)

    def test_start_grabbing_failure_restores(self):
        fake = FakeSdk(codes={"IMV_StartGrabbing": [-101]})
        result = _sweep(fake)
        self.assertEqual(result.status, "START_GRABBING_FAILED")
        self.assertNotIn("IMV_StopGrabbing", fake.names())
        self.assert_restored(fake, result)

    def test_frame_lifecycle_failure_stops_before_second_frame(self):
        for name in ("IMV_ReleaseFrame", "IMV_StopGrabbing"):
            with self.subTest(name=name):
                fake = FakeSdk(codes={name: [-101]})
                code, out, _, _ = _run(fake)
                self.assertEqual(code, es.EXIT_CLEANUP_FAILED)
                self.assertIn("SWEEP_FAILURE: FRAME_LIFECYCLE_FAILED", out)
                self.assertIn(f"Lifecycle {name} (frame 1): FAILED with code -101", out)
                self.assertEqual(fake.names().count("IMV_GetFrame"), 1)
                self.assertEqual(fake.names().count("IMV_ReleaseFrame"), 1)
                self.assertIn("Exposure restoration: RESTORED_AND_VERIFIED", out)

    def test_keyboard_interrupt_during_frame_restores(self):
        fake = FakeSdk(raises={"IMV_GetFrame": [KeyboardInterrupt()]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_INTERRUPTED)
        self.assertIn("SWEEP_FAILURE: INTERRUPTED", out)
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertIn("Exposure restoration: RESTORED_AND_VERIFIED", out)
        self.assertEqual(fake.handles_alive, 0)


# 22-24. Restoration and cleanup failures are visible and prevent success.
class RestoreAndCleanupFailureTests(NoPhysicalSdkTestCase):
    def test_restore_write_failure_prevents_success(self):
        fake = FakeSdk(codes={WRITE: [0, 0, -101]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_RESTORE_FAILED)
        self.assertIn("Exposure restoration: RESTORE_WRITE_FAILED (original 5000.0, write code -101)", out)
        self.assertIn("RESTORE_FAILURE", out)
        self.assertIn("SWEEP_STATUS: FAILED", out)
        self.assertNotIn("SWEEP_STATUS: SUCCESS", out)
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])

    def test_restore_exception_prevents_success(self):
        fake = FakeSdk(raises={WRITE: [None, None, OSError("boom")]})
        result = _sweep(fake)
        self.assertEqual(result.status, es.STATUS_COMPLETE)
        self.assertFalse(result.ok)
        self.assertEqual(result.restore.status, es.RESTORE_WRITE_FAILED)
        self.assertIn("boom", result.restore.error)
        self.assertEqual(es.exit_code_for(result), es.EXIT_RESTORE_FAILED)
        self.assertEqual(fake.handles_alive, 0)

    def test_restore_readback_verified(self):
        fake = FakeSdk(apply_set=lambda value, index: 4999.0 if index == 2 else value)
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_RESTORE_FAILED)
        self.assertIn("RESTORE_READBACK_MISMATCH (original 5000.0, read back 4999.0)", out)
        self.assertIn("SWEEP_STATUS: FAILED", out)

    def test_restore_readback_failure_prevents_success(self):
        fake = FakeSdk(codes={READ_VALUE: [0, 0, 0, -101]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_RESTORE_FAILED)
        self.assertIn("RESTORE_READBACK_FAILED", out)

    def test_restore_failure_preserves_original_error(self):
        fake = FakeSdk(codes={"IMV_GetFrame": [-119], WRITE: [0, -101]})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_RESTORE_FAILED)
        self.assertIn("SWEEP_FAILURE: GET_FRAME_FAILED", out)
        self.assertIn("RESTORE_WRITE_FAILED", out)

    def test_cleanup_failures_visible_and_prevent_success(self):
        for name in ("IMV_Close", "IMV_DestroyHandle"):
            with self.subTest(name=name):
                fake = FakeSdk(codes={name: -101})
                code, out, _, _ = _run(fake)
                self.assertEqual(code, es.EXIT_CLEANUP_FAILED)
                self.assertIn(f"Lifecycle {name}: FAILED with code -101", out)
                self.assertIn("CLEANUP_FAILURE", out)
                self.assertIn("SWEEP_STATUS: FAILED", out)
                self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")

    def test_cleanup_exception_visible_and_destroy_still_runs(self):
        fake = FakeSdk(raises={"IMV_Close": OSError("boom")})
        code, out, _, _ = _run(fake)
        self.assertEqual(code, es.EXIT_CLEANUP_FAILED)
        self.assertIn("Lifecycle IMV_Close: FAILED (OSError('boom'))", out)
        self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")
        self.assertEqual(fake.handles_alive, 0)

    def test_open_failure_destroys_handle_without_write(self):
        fake = FakeSdk(codes={"IMV_Open": -102})
        result = _sweep(fake)
        self.assertEqual(result.status, "OPEN_FAILED")
        self.assertEqual(fake.names(), ["IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open", "IMV_DestroyHandle"])
        self.assertEqual(result.restore.status, es.RESTORE_NOT_REQUIRED)


# 25. No third acquisition or retry in any scenario.
class NoRetryTests(NoPhysicalSdkTestCase):
    def test_never_more_than_two_frames_or_retries(self):
        scenarios = (
            {}, {"codes": {"IMV_GetFrame": [-119]}}, {"codes": {"IMV_GetFrame": [0, -119]}},
            {"codes": {WRITE: [-101]}}, {"codes": {"IMV_StartGrabbing": [0, -101]}},
            {"pixel_formats": (MONO10, MONO8)}, {"codes": {WRITE: [0, 0, -101]}},
        )
        for kwargs in scenarios:
            with self.subTest(kwargs=kwargs):
                fake = FakeSdk(**kwargs)
                _sweep(fake)
                self.assertLessEqual(fake.names().count("IMV_GetFrame"), 2)
                self.assertLessEqual(fake.names().count("IMV_StartGrabbing"), 2)
                self.assertLessEqual(len(fake.writes()), 3)
                self.assertEqual(fake.names().count("IMV_EnumDevices"), 1)
                self.assertEqual(fake.names().count("IMV_Open"), 1)
                self.assertEqual(fake.violations, [])

    def test_failed_get_frame_not_retried(self):
        fake = FakeSdk(codes={"IMV_GetFrame": [-119]})
        _sweep(fake)
        self.assertEqual(fake.names().count("IMV_GetFrame"), 1)
        self.assertEqual(fake.names().count("IMV_StartGrabbing"), 1)


# 26-27. No MCP, LLM, credential or hardware access.
class IsolationTests(NoPhysicalSdkTestCase):
    def test_no_mcp_or_llm_modules_loaded(self):
        before = set(sys.modules)
        with mock.patch("os.getenv", side_effect=AssertionError("environment read")):
            code, _, _, _ = _run(FakeSdk())
        self.assertEqual(code, es.EXIT_OK)
        new_modules = set(sys.modules) - before
        for name in new_modules:
            self.assertNotRegex(name, r"openai|mcp|llm|anthropic")


class SourceSafetyTests(unittest.TestCase):
    def test_source_writes_only_exposure_and_has_no_extra_activity(self):
        source = (HW3_DIR / "irayple_exposure_sweep.py").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r"IMV_Set\w*Feature\w*", source)), {"IMV_SetDoubleFeatureValue"})
        write_targets = re.findall(r"camera\.IMV_SetDoubleFeatureValue\(([^,]+),", source)
        self.assertEqual(write_targets, ["EXPOSURE_FEATURE", "EXPOSURE_FEATURE"])  # sweep write, restore write
        for forbidden in ("ExecuteCommand", "StartGrabbingEx", "AttachGrabbing", "interfaceTypeAll",
                          "modeByIndex", "OpenEx", "threading", "subprocess", "environ", "getenv",
                          "open(", "SaveImage", "write_text", "write_bytes", "while ", "retry",
                          "mcp", "openai", "api_key", "socket", "requests"):
            self.assertNotIn(forbidden, source)
        for unchanged in ("Gain\"", "PixelFormat", "Width\"", "Height\"", "OffsetX", "OffsetY",
                          "AcquisitionFrameRate"):
            self.assertNotIn(unchanged, source)


if __name__ == "__main__":
    unittest.main()
