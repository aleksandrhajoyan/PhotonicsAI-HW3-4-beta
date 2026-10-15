"""Offline tests for IRaypleCamera. Every SDK call goes to a fake binding; no hardware, network or LLM."""

import ast
import builtins
import ctypes
import importlib
import inspect
import os
import re
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

import irayple_camera as ic  # noqa: E402
from authorized_camera import AuthorizedCameraRuntime  # noqa: E402
from camera_contract import CameraBackend, CameraFrame  # noqa: E402
from session_policy import SessionEnvelope, SessionGate, session_approval  # noqa: E402

FULL_SERIAL = "DA1234567890"
FULL_CAMERA_KEY = "Huaray Technology:" + FULL_SERIAL
MONO8 = 0x01080001
MONO10 = 0x01100003
NAN = float("nan")
# 4 x 2 Mono8 frames. After release the fake SDK overwrites its buffer with 0xEE,
# so any read of SDK memory after ReleaseFrame would show up as 238 DN.
PAYLOAD_1 = bytes([0, 10, 20, 30, 40, 50, 60, 70])
PAYLOAD_2 = bytes([0, 50, 100, 150, 200, 250, 255, 255])
PAYLOAD_3 = bytes([1, 2, 3, 4, 5, 6, 7, 8])

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
OPEN_SEQUENCE = [
    "IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open",
    *["IMV_FeatureIsAvailable", "IMV_FeatureIsReadable", "IMV_GetEnumFeatureSymbol"] * 3,
    READ_VALUE, "IMV_GetDoubleFeatureMin", "IMV_GetDoubleFeatureMax",
]
CAPTURE_SEQUENCE = ["IMV_StartGrabbing", "IMV_GetFrame", "memmove", "IMV_ReleaseFrame", "IMV_StopGrabbing"]


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
    block_ids: native blockId per delivered frame; by default always 1, as the real camera reported.
    """

    def __init__(self, devices=None, codes=None, raises=None, enums=None, available=None, readable=None,
                 writeable=True, exposure=(5000.0, 16.0, 1000000.0), apply_set=None,
                 payloads=(PAYLOAD_1, PAYLOAD_2, PAYLOAD_3), width=4, height=2, pixel_format=MONO8,
                 padding=(0, 0), status=0, block_ids=None, size_delta=0, on_get_frame=None):
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
        self.pixel_format = pixel_format
        self.padding = padding
        self.status = status
        self.block_ids = block_ids
        self.size_delta = size_delta
        self.on_get_frame = on_get_frame
        self.counts = {}
        self.calls = []
        self.handles_alive = 0
        self.is_open = False
        self.is_grabbing = False
        self.frames_served = 0
        self.frames_outstanding = {}
        self.released_buffers = []
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
        if name != "ExposureTime" or type(value) is not float or self.is_grabbing or not self.is_open:
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
        if self.frames_outstanding:
            self.violations.append("GetFrame with a frame still outstanding")
        index = self.frames_served
        self.frames_served += 1
        payload = self.payloads[index % len(self.payloads)]
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        frame.pData = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        info = frame.frameInfo
        info.blockId = 1 if self.block_ids is None else self.block_ids[index]
        info.status = self.status
        info.width, info.height = self.width, self.height
        info.size = len(payload) + self.size_delta
        info.pixelFormat = self.pixel_format
        info.paddingX, info.paddingY = self.padding
        self.frames_outstanding[ctypes.addressof(frame)] = buffer
        self.exposure_at_frame.append(self.current)
        if self.on_get_frame is not None:
            self.on_get_frame()

    def _on_IMV_ReleaseFrame(self, frame):
        buffer = self.frames_outstanding.pop(ctypes.addressof(frame), None)
        if buffer is None:
            self.violations.append("release of a frame not outstanding")
            return
        # The SDK may reuse its buffer after release; overwrite it to prove nothing still reads it.
        ctypes.memset(buffer, 0xEE, len(buffer))
        self.released_buffers.append(buffer)

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


class OfflineTestCase(unittest.TestCase):
    """Any attempt to load the real SDK or change directory fails loudly; frame copies are logged."""

    def setUp(self):
        self.fake = None
        guard = AssertionError("real SDK access attempted in an offline test")
        real_memmove = ctypes.memmove

        def logged_memmove(dst, src, size):
            if self.fake is not None:
                self.fake.calls.append(("memmove", size))
            return real_memmove(dst, src, size)

        for target in (
            mock.patch("ctypes.WinDLL", side_effect=guard),
            mock.patch("ctypes.CDLL", side_effect=guard),
            mock.patch("os.add_dll_directory", side_effect=guard, create=True),
            mock.patch("os.chdir", side_effect=AssertionError("working directory changed")),
            mock.patch.object(ic.discovery, "load_vendor_sdk", side_effect=guard),
            mock.patch.object(ic.ctypes, "memmove", side_effect=logged_memmove),
        ):
            target.start()
            self.addCleanup(target.stop)

    def tearDown(self):
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)
        if self.fake is not None:
            self.assertEqual(self.fake.violations, [])

    def make(self, **kwargs):
        self.fake = FakeSdk(**kwargs)
        return self.fake, ic.IRaypleCamera(self.fake.ns)

    def opened(self, **kwargs):
        fake, camera = self.make(**kwargs)
        camera.open()
        return fake, camera

    def calls_since(self, fake, mark):
        return fake.names()[mark:]

    def assert_shut_down(self, fake):
        self.assertFalse(fake.is_grabbing)
        self.assertFalse(fake.is_open)
        self.assertEqual(fake.handles_alive, 0)
        self.assertEqual(fake.frames_outstanding, {})


# 1, 34, 35. Import safety.
class ImportSafetyTests(unittest.TestCase):
    def test_import_loads_no_sdk_and_keeps_working_directory(self):
        cwd = os.getcwd()
        with mock.patch.object(ic.discovery, "load_vendor_sdk") as loader, \
                mock.patch("ctypes.WinDLL") as win_dll, \
                mock.patch("ctypes.CDLL") as cdll, \
                mock.patch("os.add_dll_directory", create=True) as add_dir, \
                mock.patch("os.chdir") as chdir, \
                mock.patch.object(importlib, "import_module") as import_module:
            importlib.reload(ic)
        for patched in (win_dll, cdll, add_dir, chdir, import_module, loader):
            patched.assert_not_called()
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn("IMVApi", sys.modules)
        self.assertNotIn("IMVDefines", sys.modules)


# 2, 3. Construction and dependency injection.
class ConstructionTests(OfflineTestCase):
    def test_constructor_performs_no_camera_operations(self):
        fake, camera = self.make()
        self.assertEqual(fake.calls, [])
        self.assertEqual(fake.handles_alive, 0)
        self.assertFalse(camera.is_open)
        info = camera.get_info()
        self.assertEqual(fake.calls, [])
        self.assertEqual((info["source"], info["synthetic"], info["is_open"]), ("hardware", False, False))
        self.assertIsNone(info["last_native_block_id"])
        self.assertEqual(info["frames_acquired"], 0)

    def test_sdk_dependency_is_injected(self):
        self.assertEqual(list(inspect.signature(ic.IRaypleCamera).parameters), ["sdk"])
        for bad in (None, object(), "IMVApi", SimpleNamespace(MvCamera=object)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                ic.IRaypleCamera(bad)
        self.make()
        ic.discovery.load_vendor_sdk.assert_not_called()

    def test_satisfies_camera_backend_protocol(self):
        _, camera = self.make()
        self.assertIsInstance(camera, CameraBackend)

    def test_closed_backend_refuses_without_sdk_calls(self):
        fake, camera = self.make()
        for call in (camera.capture_frame, lambda: camera.set_exposure_us(1000.0), camera.get_exposure_limits_us):
            with self.assertRaises(RuntimeError):
                call()
        self.assertEqual(fake.calls, [])

    def test_preload_uses_existing_loader_once(self):
        sentinel = SimpleNamespace()
        with mock.patch.object(ic.discovery, "load_vendor_sdk", return_value=sentinel) as loader:
            self.assertIs(ic.preload_vendor_sdk(), sentinel)
        loader.assert_called_once_with()


# 4-10, 32. Open lifecycle.
class OpenTests(OfflineTestCase):
    def test_open_selects_expected_usb3_camera_by_key(self):
        fake, camera = self.opened()
        self.assertEqual(fake.names(), OPEN_SEQUENCE)
        self.assertEqual(fake.calls[0][2], 0x2)  # interfaceTypeUsb3 only
        create = fake.calls[1]
        self.assertEqual(create[1], 1)  # modeByCameraKey
        self.assertEqual(create[2].value, FULL_CAMERA_KEY.encode())
        reads = [call[1] for call in fake.calls if call[0] == "IMV_GetEnumFeatureSymbol"]
        self.assertEqual(reads, ["TriggerMode", "ExposureAuto", "GainAuto"])
        self.assertTrue(camera.is_open)
        info = camera.get_info()
        self.assertEqual(info["model"], "MV-A7200MU130")
        self.assertEqual(info["serial"], "****7890")
        self.assertNotIn(FULL_SERIAL, repr(info))

    def test_wrong_or_multiple_devices_refuse(self):
        cases = (
            ([], "NO_DEVICE"),
            ([_device(), _device(serial=b"DA0000001111")], "MULTIPLE_DEVICES"),
            ([_device(model=b"MV-OTHER")], "UNEXPECTED_DEVICE"),
            ([_device(camera_type=0, interface_type=1)], "UNEXPECTED_TRANSPORT"),
        )
        for devices, status in cases:
            with self.subTest(status=status):
                fake, camera = self.make(devices=devices)
                with self.assertRaises(ic.IRaypleCameraError) as ctx:
                    camera.open()
                self.assertEqual(ctx.exception.status, status)
                self.assertEqual(fake.names(), ["IMV_EnumDevices"])
                self.assertFalse(camera.is_open)

    def test_open_uses_one_handle(self):
        fake, camera = self.opened()
        camera.open()  # already open: no further SDK call
        self.assertEqual(fake.names(), OPEN_SEQUENCE)
        self.assertEqual(fake.handles_alive, 1)
        self.assertEqual(fake.counts["IMV_CreateHandle"], 1)

    def _assert_refused(self, fake, camera, status):
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.open()
        self.assertEqual(ctx.exception.status, status)
        self.assertIsNone(ctx.exception.cleanup_error)
        self.assertFalse(camera.is_open)
        self.assertEqual(fake.writes(), [])
        self.assertNotIn("IMV_StartGrabbing", fake.names())
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assert_shut_down(fake)

    def test_exposure_auto_and_gain_auto_must_be_off(self):
        for enums, status in (({"ExposureAuto": b"Continuous"}, "EXPOSURE_AUTO_NOT_OFF"),
                              ({"ExposureAuto": b"Once"}, "EXPOSURE_AUTO_NOT_OFF"),
                              ({"GainAuto": b"Continuous"}, "GAIN_AUTO_NOT_OFF")):
            with self.subTest(enums=enums):
                self._assert_refused(*self.make(enums=enums), status)

    def test_trigger_mode_must_be_off(self):
        self._assert_refused(*self.make(enums={"TriggerMode": b"On"}), "TRIGGER_MODE_NOT_OFF")

    def test_unavailable_or_unreadable_features_refuse(self):
        for kwargs in ({"available": {"TriggerMode", "ExposureAuto"}}, {"readable": {"TriggerMode", "GainAuto"}},
                       {"codes": {"IMV_GetEnumFeatureSymbol": [0, -111]}}):
            with self.subTest(kwargs=kwargs):
                self._assert_refused(*self.make(**kwargs), "FEATURE_UNAVAILABLE")
        self._assert_refused(*self.make(enums={"GainAuto": b""}), "GAIN_AUTO_NOT_OFF")

    def test_open_modifies_no_feature_and_does_not_grab(self):
        fake, _ = self.opened()
        for name in (WRITE, "IMV_StartGrabbing", "IMV_GetFrame", *FORBIDDEN_CALLS):
            self.assertNotIn(name, fake.names())
        self.assertEqual(fake.current, 5000.0)

    def test_exposure_limits_equal_sdk_reported_limits(self):
        for exposure in ((5000.0, 16.0, 1000000.0), (300.25, 20.5, 200000.0)):
            with self.subTest(exposure=exposure):
                fake, camera = self.opened(exposure=exposure)
                mark = len(fake.calls)
                self.assertEqual(camera.get_exposure_limits_us(), exposure[1:])
                self.assertEqual(camera.get_info()["exposure_limits_us"], list(exposure[1:]))
                self.assertEqual(camera.get_info()["exposure_us"], exposure[0])
                self.assertEqual(len(fake.calls), mark)  # cached, no SDK call

    def test_invalid_sdk_exposure_values_refuse_open(self):
        for exposure in ((NAN, 16.0, 1e6), (5000.0, NAN, 1e6), (5000.0, 16.0, 16.0), (5000.0, 16.0, 4000.0),
                         (5000.0, -1.0, 1e6)):
            with self.subTest(exposure=exposure):
                fake, camera = self.make(exposure=exposure)
                with self.assertRaises(ic.IRaypleCameraError):
                    camera.open()
                self.assert_shut_down(fake)
                self.assertFalse(camera.is_open)

    def test_partial_open_failure_cleans_up(self):
        cases = (
            ({"codes": {"IMV_CreateHandle": -101}}, ["IMV_EnumDevices", "IMV_CreateHandle"]),
            ({"codes": {"IMV_Open": -102}}, ["IMV_EnumDevices", "IMV_CreateHandle", "IMV_Open",
                                              "IMV_DestroyHandle"]),
            ({"codes": {"IMV_GetDoubleFeatureMax": -111}}, OPEN_SEQUENCE + ["IMV_Close", "IMV_DestroyHandle"]),
            ({"raises": {"IMV_GetDoubleFeatureMin": OSError("boom")}},
             OPEN_SEQUENCE[:-1] + ["IMV_Close", "IMV_DestroyHandle"]),
        )
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                fake, camera = self.make(**kwargs)
                with self.assertRaises(Exception):
                    camera.open()
                self.assertEqual(fake.names(), expected)
                self.assert_shut_down(fake)
                self.assertFalse(camera.is_open)

    def test_open_failure_native_code_preserved(self):
        fake, camera = self.make(codes={"IMV_Open": -102})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.open()
        self.assertEqual((ctx.exception.status, ctx.exception.sdk_code, ctx.exception.sdk_code_name),
                         ("OPEN_FAILED", -102, "IMV_INVALID_HANDLE"))

    def test_reopen_after_clean_open_failure(self):
        fake, camera = self.make(codes={"IMV_Open": [-102]})
        with self.assertRaises(ic.IRaypleCameraError):
            camera.open()
        camera.open()
        self.assertTrue(camera.is_open)
        self.assertEqual(fake.handles_alive, 1)

    def test_partial_open_cleanup_failure_is_visible(self):
        fake, camera = self.make(enums={"TriggerMode": b"On"}, codes={"IMV_DestroyHandle": -101})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.open()
        self.assertEqual(ctx.exception.status, "TRIGGER_MODE_NOT_OFF")
        self.assertIsInstance(ctx.exception.cleanup_error, ic.IRaypleCloseError)
        self.assertEqual(ctx.exception.cleanup_error.status, ic.STATUS_CLEANUP_FAILED)
        self.assertIn("IMV_DestroyHandle FAILED with code -101", " ".join(ctx.exception.__notes__))
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCloseError):
            camera.close()
        with self.assertRaises(ic.IRaypleCameraError) as reopen:
            camera.open()
        self.assertEqual(reopen.exception.status, "UNRESOLVED_CLEANUP_FAILURE")
        self.assertEqual(len(fake.calls), mark)


# 11-14. Exposure.
class ExposureTests(OfflineTestCase):
    def test_valid_exposure_written_and_read_back(self):
        fake, camera = self.opened()
        mark = len(fake.calls)
        self.assertEqual(camera.set_exposure_us(1000), 1000.0)
        names = self.calls_since(fake, mark)
        self.assertEqual(names, [*["IMV_FeatureIsAvailable", "IMV_FeatureIsReadable",
                                   "IMV_GetEnumFeatureSymbol"] * 3,
                                 "IMV_FeatureIsWriteable", WRITE, READ_VALUE])
        self.assertEqual(fake.calls[-2], (WRITE, "ExposureTime", 1000.0))
        self.assertIs(type(fake.calls[-2][2]), float)
        self.assertEqual(camera.get_info()["exposure_us"], 1000.0)
        self.assertTrue(camera.get_info()["exposure_restore_required"])
        self.assertEqual(camera.capture_frame().exposure_us, 1000.0)

    def test_limits_are_inclusive(self):
        fake, camera = self.opened()
        self.assertEqual(camera.set_exposure_us(16.0), 16.0)
        self.assertEqual(camera.set_exposure_us(1000000.0), 1000000.0)

    def test_invalid_or_out_of_range_rejected_without_sdk_calls(self):
        fake, camera = self.opened()
        mark = len(fake.calls)
        for value, error in ((True, TypeError), (False, TypeError), ("1000", TypeError), (None, TypeError),
                             (b"1000", TypeError), ([1000.0], TypeError), (NAN, ValueError),
                             (float("inf"), ValueError), (float("-inf"), ValueError), (15.9, ValueError),
                             (1000000.5, ValueError), (-1000.0, ValueError), (0, ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                camera.set_exposure_us(value)
        self.assertEqual(len(fake.calls), mark)
        self.assertFalse(camera.get_info()["exposure_restore_required"])

    def test_preconditions_and_writeability_checked_before_write(self):
        fake, camera = self.opened()
        fake.enums["ExposureAuto"] = b"Continuous"
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(1000.0)
        self.assertEqual(ctx.exception.status, "EXPOSURE_AUTO_NOT_OFF")
        fake.enums["ExposureAuto"] = b"Off"
        fake.writeable = False
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(1000.0)
        self.assertEqual(ctx.exception.status, "EXPOSURE_NOT_WRITEABLE")
        self.assertEqual(fake.writes(), [])
        self.assertFalse(camera.get_info()["exposure_restore_required"])
        camera.close()
        self.assertEqual(fake.writes(), [])

    def test_write_failure_triggers_later_restoration(self):
        fake, camera = self.opened(codes={WRITE: [-111]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(1000.0)
        self.assertEqual((ctx.exception.status, ctx.exception.sdk_code, ctx.exception.sdk_code_name),
                         ("EXPOSURE_WRITE_FAILED", -111, "IMV_INVALID_ACCESS"))
        self.assertIsNone(camera.get_info()["exposure_us"])
        self.assertTrue(camera.get_info()["exposure_restore_required"])
        camera.close()
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertEqual(fake.names()[-4:], [WRITE, READ_VALUE, "IMV_Close", "IMV_DestroyHandle"])

    def test_write_exception_still_restored(self):
        fake, camera = self.opened(raises={WRITE: [OSError("access violation")]})
        with self.assertRaises(OSError):
            camera.set_exposure_us(1000.0)
        camera.close()
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertEqual(fake.current, 5000.0)

    def test_readback_mismatch_fails_closed(self):
        fake, camera = self.opened(apply_set=lambda value, index: 998.4 if index == 0 else value)
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(1000.0)
        self.assertEqual(ctx.exception.status, "EXPOSURE_READBACK_MISMATCH")
        self.assertIn("998.4", str(ctx.exception))
        self.assertEqual(fake.writes(), [1000.0])  # never retried or clamped
        with self.assertRaises(ic.IRaypleCameraError) as capture:
            camera.capture_frame()
        self.assertEqual(capture.exception.status, "EXPOSURE_UNVERIFIED")
        self.assertNotIn("IMV_GetFrame", fake.names())
        camera.close()
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertEqual(fake.current, 5000.0)

    def test_silent_rounding_rejected(self):
        fake, camera = self.opened(apply_set=lambda value, index: float(np.float32(value)))
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(16.3)
        self.assertEqual(ctx.exception.status, "EXPOSURE_READBACK_MISMATCH")

    def test_readback_failure_fails_closed(self):
        fake, camera = self.opened(codes={READ_VALUE: [0, -101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.set_exposure_us(1000.0)
        self.assertEqual(ctx.exception.status, "EXPOSURE_READBACK_FAILED")
        self.assertIsNone(camera.get_info()["exposure_us"])

    def test_successful_write_after_failure_reverifies(self):
        fake, camera = self.opened(codes={WRITE: [-111]})
        with self.assertRaises(ic.IRaypleCameraError):
            camera.set_exposure_us(1000.0)
        self.assertEqual(camera.set_exposure_us(2000.0), 2000.0)
        self.assertEqual(camera.capture_frame().exposure_us, 2000.0)


# 15-28. Acquisition.
class CaptureTests(OfflineTestCase):
    def test_exactly_one_get_frame_per_capture(self):
        fake, camera = self.opened()
        mark = len(fake.calls)
        camera.capture_frame()
        self.assertEqual(self.calls_since(fake, mark), CAPTURE_SEQUENCE)
        get_frame = [call for call in fake.calls if call[0] == "IMV_GetFrame"]
        self.assertEqual(len(get_frame), 1)
        self.assertEqual(get_frame[0][2], ic.GET_FRAME_TIMEOUT_MS)
        self.assertEqual(fake.calls[mark + 2], ("memmove", 8))
        self.assertEqual(fake.frames_outstanding, {})
        self.assertFalse(fake.is_grabbing)

    def test_mono8_decoding_and_shape(self):
        fake, camera = self.opened()
        frame = camera.capture_frame()
        self.assertIsInstance(frame, CameraFrame)
        self.assertEqual(frame.pixels.dtype, np.uint8)
        self.assertEqual(frame.pixels.shape, (2, 4))
        np.testing.assert_array_equal(frame.pixels, np.frombuffer(PAYLOAD_1, np.uint8).reshape(2, 4))
        self.assertEqual((frame.bit_depth, frame.source, frame.exposure_us), (8, "hardware", 5000.0))

    def test_array_owns_its_memory(self):
        _, camera = self.opened()
        pixels = camera.capture_frame().pixels
        self.assertTrue(pixels.flags.owndata)
        self.assertIsNone(pixels.base)
        self.assertTrue(pixels.flags.c_contiguous)

    def test_pixels_remain_valid_after_vendor_release(self):
        fake, camera = self.opened()
        frame = camera.capture_frame()
        self.assertEqual(len(fake.released_buffers), 1)
        self.assertEqual(bytes(fake.released_buffers[0]), b"\xee" * 8)  # SDK buffer reused after release
        self.assertEqual(frame.pixels.tobytes(), PAYLOAD_1)

    def test_release_and_stop_precede_frame_construction(self):
        fake, camera = self.opened()

        def record(**fields):
            fake.calls.append(("CameraFrame",))
            return CameraFrame(**fields)

        mark = len(fake.calls)
        with mock.patch.object(ic, "CameraFrame", side_effect=record):
            camera.capture_frame()
        self.assertEqual(self.calls_since(fake, mark), CAPTURE_SEQUENCE + ["CameraFrame"])

    def test_frame_metadata_validated(self):
        cases = (
            ({"status": 1}, "INVALID_FRAME"),
            ({"pixel_format": MONO10, "size_delta": 8}, "NOT_MONO8"),  # 16 bytes: 4x2 at 16 occupied bits
            ({"pixel_format": MONO10}, "INVALID_FRAME"),  # 8 bytes: too small for 4x2 Mono10
            ({"padding": (2, 0)}, "INVALID_LAYOUT"),
            ({"padding": (0, 1)}, "INVALID_LAYOUT"),
            ({"size_delta": 4}, "INVALID_LAYOUT"),
            ({"size_delta": -1}, "INVALID_FRAME"),
            ({"width": 0}, "INVALID_FRAME"),
        )
        for kwargs, status in cases:
            with self.subTest(kwargs=kwargs):
                fake, camera = self.opened(**kwargs)
                with self.assertRaises(ic.IRaypleCameraError) as ctx:
                    camera.capture_frame()
                self.assertEqual(ctx.exception.status, status)
                self.assertNotIn("memmove", fake.names())
                self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)
                self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
                self.assertEqual(fake.frames_outstanding, {})
                self.assertEqual(camera.get_info()["frames_acquired"], 0)
                self.assertIsNone(camera.get_info()["last_native_block_id"])

    def test_native_block_id_one_twice_yields_frame_ids_one_and_two(self):
        fake, camera = self.opened(block_ids=[1, 1])
        ids = [camera.capture_frame().frame_id for _ in range(2)]
        self.assertEqual(ids, [1, 2])
        self.assertEqual(camera.get_info()["last_native_block_id"], 1)

    def test_three_captures_yield_one_two_three(self):
        _, camera = self.opened()
        self.assertEqual([camera.capture_frame().frame_id for _ in range(3)], [1, 2, 3])

    def test_failed_acquisition_does_not_advance_counter(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": [0, -119]})
        self.assertEqual(camera.capture_frame().frame_id, 1)
        with self.assertRaises(ic.IRaypleCameraError):
            camera.capture_frame()
        self.assertEqual(camera.capture_frame().frame_id, 2)
        self.assertEqual(camera.get_info()["frames_acquired"], 2)

    def test_grabbing_restarts_and_reopen_do_not_reset_frame_id(self):
        fake, camera = self.opened()
        self.assertEqual([camera.capture_frame().frame_id for _ in range(2)], [1, 2])
        camera.close()
        camera.open()
        self.assertEqual(camera.capture_frame().frame_id, 3)
        self.assertEqual(fake.counts["IMV_StartGrabbing"], 3)
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 3)
        self.assertEqual(ic.IRaypleCamera(fake.ns).get_info()["frames_acquired"], 0)  # per instance

    def test_native_block_id_kept_distinct_and_not_fabricated(self):
        fake, camera = self.opened(block_ids=[77, 78, 79], codes={"IMV_GetFrame": [0, 0, -119]})
        self.assertIsNone(camera.get_info()["last_native_block_id"])
        frames = [camera.capture_frame() for _ in range(2)]
        self.assertEqual([frame.frame_id for frame in frames], [1, 2])
        self.assertEqual(camera.get_info()["last_native_block_id"], 78)
        with self.assertRaises(ic.IRaypleCameraError):
            camera.capture_frame()
        self.assertEqual(camera.get_info()["last_native_block_id"], 78)
        self.assertIn("not a camera-native identifier", camera.get_info()["frame_id_basis"])

    def test_captures_return_independent_arrays(self):
        _, camera = self.opened()
        first, second = camera.capture_frame(), camera.capture_frame()
        self.assertFalse(np.shares_memory(first.pixels, second.pixels))
        self.assertEqual(first.pixels.tobytes(), PAYLOAD_1)
        self.assertEqual(second.pixels.tobytes(), PAYLOAD_2)
        second.pixels[0, 0] = 99
        self.assertEqual(first.pixels.tobytes(), PAYLOAD_1)

    def test_get_frame_failure_stops_grabbing_without_release(self):
        fake, camera = self.opened(codes={"IMV_GetFrame": -119})
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.capture_frame()
        self.assertEqual((ctx.exception.status, ctx.exception.sdk_code_name), ("GET_FRAME_FAILED", "IMV_TIMEOUT"))
        self.assertEqual(self.calls_since(fake, mark), ["IMV_StartGrabbing", "IMV_GetFrame", "IMV_StopGrabbing"])
        self.assertFalse(fake.is_grabbing)
        self.assertTrue(camera.is_open)

    def test_start_grabbing_failure_does_not_get_frame(self):
        fake, camera = self.opened(codes={"IMV_StartGrabbing": -101})
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.capture_frame()
        self.assertEqual(ctx.exception.status, "START_GRABBING_FAILED")
        self.assertEqual(self.calls_since(fake, mark), ["IMV_StartGrabbing"])

    def test_release_failure_is_not_success(self):
        fake, camera = self.opened(codes={"IMV_ReleaseFrame": [-101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.capture_frame()
        self.assertEqual(ctx.exception.status, "FRAME_LIFECYCLE_FAILED")
        self.assertIn("IMV_ReleaseFrame FAILED with code -101", str(ctx.exception))
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)  # released exactly once, never retried
        self.assertEqual(fake.counts["IMV_StopGrabbing"], 1)
        self.assertEqual(camera.get_info()["frames_acquired"], 0)
        fake.frames_outstanding.clear()  # the fake kept the frame because its release "failed"
        camera.close()
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)

    def test_stop_failure_is_not_success_and_blocks_further_use(self):
        fake, camera = self.opened(codes={"IMV_StopGrabbing": [-101]})
        with self.assertRaises(ic.IRaypleCameraError) as ctx:
            camera.capture_frame()
        self.assertEqual(ctx.exception.status, "FRAME_LIFECYCLE_FAILED")
        self.assertEqual(camera.get_info()["frames_acquired"], 0)
        mark = len(fake.calls)
        for call in (camera.capture_frame, lambda: camera.set_exposure_us(1000.0)):
            with self.assertRaises(ic.IRaypleCameraError) as blocked:
                call()
            self.assertEqual(blocked.exception.status, "ACQUISITION_ACTIVE")
        self.assertEqual(len(fake.calls), mark)
        camera.close()  # stop is attempted again during shutdown
        self.assertEqual(fake.names()[mark:], ["IMV_StopGrabbing", "IMV_Close", "IMV_DestroyHandle"])
        self.assert_shut_down(fake)

    def test_exception_during_copy_still_releases(self):
        fake, camera = self.opened()
        with mock.patch.object(ic.ctypes, "memmove", side_effect=MemoryError("copy failed")):
            with self.assertRaises(MemoryError):
                camera.capture_frame()
        self.assertEqual(fake.counts["IMV_ReleaseFrame"], 1)
        self.assertFalse(fake.is_grabbing)
        self.assertEqual(fake.frames_outstanding, {})

    def test_no_automatic_retry(self):
        scenarios = ({"codes": {"IMV_GetFrame": -119}}, {"codes": {"IMV_StartGrabbing": -101}},
                     {"status": 3}, {"pixel_format": MONO10}, {"codes": {"IMV_ReleaseFrame": [-101]}})
        for kwargs in scenarios:
            with self.subTest(kwargs=kwargs):
                fake, camera = self.opened(**kwargs)
                with self.assertRaises(ic.IRaypleCameraError):
                    camera.capture_frame()
                self.assertLessEqual(fake.counts.get("IMV_GetFrame", 0), 1)
                self.assertEqual(fake.counts["IMV_StartGrabbing"], 1)
                self.assertEqual(fake.counts["IMV_EnumDevices"], 1)
                fake.frames_outstanding.clear()


# 29-33. Close and restoration.
class CloseTests(OfflineTestCase):
    def test_close_without_writes_skips_restoration(self):
        fake, camera = self.opened()
        camera.capture_frame()
        mark = len(fake.calls)
        camera.close()
        self.assertEqual(self.calls_since(fake, mark), ["IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.writes(), [])
        self.assert_shut_down(fake)

    def test_close_restores_original_exposure_and_verifies(self):
        fake, camera = self.opened()
        camera.set_exposure_us(1000.0)
        camera.capture_frame()
        mark = len(fake.calls)
        camera.close()
        self.assertEqual(self.calls_since(fake, mark), [WRITE, READ_VALUE, "IMV_Close", "IMV_DestroyHandle"])
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assertEqual(fake.current, 5000.0)
        self.assertEqual(fake.exposure_at_frame, [1000.0])
        self.assertFalse(camera.is_open)

    def test_restore_write_failure_visible(self):
        fake, camera = self.opened(codes={WRITE: [0, -101]})
        camera.set_exposure_us(1000.0)
        with self.assertRaises(ic.IRaypleCloseError) as ctx:
            camera.close()
        error = ctx.exception
        self.assertEqual(error.status, ic.STATUS_RESTORE_FAILED)
        self.assertEqual(error.restore.status, ic.RESTORE_WRITE_FAILED)
        self.assertFalse(error.restore_ok)
        self.assertTrue(error.cleanup_ok)
        self.assertIn("write code -101", str(error))
        self.assertEqual(fake.names()[-2:], ["IMV_Close", "IMV_DestroyHandle"])
        self.assert_shut_down(fake)

    def test_restore_readback_mismatch_and_failure_visible(self):
        for kwargs, status in (({"apply_set": lambda value, index: 4999.0 if index == 1 else value},
                                ic.RESTORE_MISMATCH),
                               ({"codes": {READ_VALUE: [0, 0, -101]}}, ic.RESTORE_READBACK_FAILED),
                               ({"raises": {WRITE: [None, OSError("boom")]}}, ic.RESTORE_WRITE_FAILED)):
            with self.subTest(status=status):
                fake, camera = self.opened(**kwargs)
                camera.set_exposure_us(1000.0)
                with self.assertRaises(ic.IRaypleCloseError) as ctx:
                    camera.close()
                self.assertEqual(ctx.exception.status, ic.STATUS_RESTORE_FAILED)
                self.assertEqual(ctx.exception.restore.status, status)
                self.assertEqual(fake.handles_alive, 0)

    def test_cleanup_failures_reported_distinctly(self):
        for name in ("IMV_Close", "IMV_DestroyHandle"):
            with self.subTest(name=name):
                fake, camera = self.opened(codes={name: -101})
                camera.set_exposure_us(1000.0)
                with self.assertRaises(ic.IRaypleCloseError) as ctx:
                    camera.close()
                self.assertEqual(ctx.exception.status, ic.STATUS_CLEANUP_FAILED)
                self.assertTrue(ctx.exception.restore_ok)
                self.assertFalse(ctx.exception.cleanup_ok)
                self.assertIn(f"{name} FAILED with code -101", str(ctx.exception))
                self.assertEqual(fake.names()[-1], "IMV_DestroyHandle")
        fake, camera = self.opened(codes={WRITE: [0, -101]}, raises={"IMV_Close": OSError("boom")})
        camera.set_exposure_us(1000.0)
        with self.assertRaises(ic.IRaypleCloseError) as ctx:
            camera.close()
        self.assertEqual(ctx.exception.status, ic.STATUS_RESTORE_AND_CLEANUP_FAILED)
        self.assertEqual(fake.handles_alive, 0)

    def test_close_and_destroy_called_once_and_close_idempotent(self):
        fake, camera = self.opened()
        camera.set_exposure_us(1000.0)
        camera.close()
        mark = len(fake.calls)
        camera.close()
        camera.close()
        self.assertEqual(len(fake.calls), mark)
        self.assertEqual(fake.counts["IMV_Close"], 1)
        self.assertEqual(fake.counts["IMV_DestroyHandle"], 1)
        self.assertEqual(fake.counts[WRITE], 2)

    def test_close_of_never_opened_backend_is_noop(self):
        fake, camera = self.make()
        camera.close()
        self.assertEqual(fake.calls, [])

    def test_failed_close_never_reported_as_success(self):
        fake, camera = self.opened(codes={"IMV_Close": -101})
        with self.assertRaises(ic.IRaypleCloseError):
            camera.close()
        mark = len(fake.calls)
        with self.assertRaises(ic.IRaypleCloseError) as again:
            camera.close()
        self.assertIn("earlier close()", str(again.exception))
        self.assertEqual(len(fake.calls), mark)  # native shutdown is not repeated
        self.assertIsNotNone(camera.get_info()["unresolved_cleanup_failure"])


# F. Sequential use only.
class ConcurrencyTests(OfflineTestCase):
    def test_reentrant_operations_rejected(self):
        rejected = []

        def reenter():
            for call in (camera.capture_frame, lambda: camera.set_exposure_us(1000.0), camera.close,
                         camera.open):
                try:
                    call()
                except RuntimeError as exc:
                    rejected.append(str(exc))

        fake, camera = self.opened(on_get_frame=reenter)
        threads_before = threading.active_count()
        frame = camera.capture_frame()
        self.assertEqual(frame.frame_id, 1)
        self.assertEqual(len(rejected), 4)
        for message in rejected:
            self.assertIn("sequential use only", message)
        self.assertEqual(fake.counts["IMV_GetFrame"], 1)
        self.assertEqual(threading.active_count(), threads_before)
        camera.close()


# 34, 35, 37. Isolation.
class IsolationTests(OfflineTestCase):
    def _full_lifecycle(self):
        fake, camera = self.opened()
        camera.get_info()
        camera.get_exposure_limits_us()
        camera.set_exposure_us(1000.0)
        camera.capture_frame()
        camera.close()
        return fake

    def test_working_directory_never_changes(self):
        cwd = os.getcwd()
        self._full_lifecycle()  # os.chdir is guarded in setUp
        self.assertEqual(os.getcwd(), cwd)

    def test_operations_do_not_import_vendor_modules(self):
        imported = []
        real_import = builtins.__import__
        real_import_module = importlib.import_module

        def spy_import(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        def spy_import_module(name, *args, **kwargs):
            imported.append(name)
            return real_import_module(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=spy_import), \
                mock.patch.object(importlib, "import_module", side_effect=spy_import_module):
            self._full_lifecycle()
        for name in imported:
            self.assertNotIn(name, ("IMVApi", "IMVDefines", "MVSDK"))
        ic.discovery.load_vendor_sdk.assert_not_called()

    def test_no_files_network_or_llm(self):
        before = set(sys.modules)
        with mock.patch.object(builtins, "open", side_effect=AssertionError("file opened")), \
                mock.patch("socket.socket", side_effect=AssertionError("network")), \
                mock.patch("os.getenv", side_effect=AssertionError("environment read")):
            self._full_lifecycle()
        for name in set(sys.modules) - before:
            self.assertNotRegex(name, r"openai|mcp|llm|anthropic|langgraph")

    def test_backend_class_source_has_no_loader_or_directory_change(self):
        source = (HW3_DIR / "irayple_camera.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IRaypleCamera")
        body = ast.get_source_segment(source, backend)
        for forbidden in ("load_vendor_sdk", "preload_vendor_sdk", "chdir", "import_module", "IMVApi\"",
                          "add_dll_directory", "WinDLL", "CDLL", "sys.path", "import "):
            self.assertNotIn(forbidden, body)
        nested_imports = [node for node in ast.walk(backend) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(nested_imports, [])
        self.assertEqual(set(re.findall(r"IMV_Set\w*", source)), {"IMV_SetDoubleFeatureValue"})
        for forbidden in ("Gain\"", "PixelFormat", "OffsetX", "TriggerSoftware", "ExecuteCommand",
                          "StartGrabbingEx", "interfaceTypeAll", "modeByIndex", "subprocess", "socket",
                          "openai", "api_key", "retry", "while ", "Thread("):
            self.assertNotIn(forbidden, source)


# 36. AuthorizedCameraRuntime accepts the hardware CameraFrame contract.
class RuntimeIntegrationTests(OfflineTestCase):
    def test_authorized_runtime_accepts_backend(self):
        tools = ("camera_info", "camera_open", "camera_set_exposure", "camera_measure", "camera_close")
        envelope = SessionEnvelope(
            session_id="session-hw", device_label="irayple-fake-sdk", allowed_tools=tools,
            min_exposure_us=100.0, max_exposure_us=10000.0, max_measurements=2,
            max_tool_calls=8, max_duration_seconds=60.0,
        )
        gate = SessionGate(envelope, lambda: 0.0)
        self.assertTrue(gate.submit_decision(session_approval(envelope.session_id, envelope.sha256(), True)))
        fake, camera = self.make()
        runtime = AuthorizedCameraRuntime(camera, gate)
        self.assertEqual(fake.calls, [])

        runtime.execute("a-1", "camera_open")
        info = runtime.execute("a-2", "camera_info")
        self.assertEqual((info["source"], info["synthetic"]), ("hardware", False))
        self.assertEqual(info["backend_exposure_limits_us"], {"min": 16.0, "max": 1000000.0})
        self.assertTrue(info["approved_range_within_backend_limits"])
        self.assertEqual(runtime.execute("a-3", "camera_set_exposure", {"exposure_us": 1000.0})["exposure_us"],
                         1000.0)
        first = runtime.execute("a-4", "camera_measure")
        second = runtime.execute("a-5", "camera_measure")
        self.assertEqual((first["source"], first["synthetic"]), ("hardware", False))
        self.assertEqual([first["frame_id"], second["frame_id"]], [1, 2])
        self.assertEqual((first["exposure_us"], first["bit_depth"], first["width"], first["height"]),
                         (1000.0, 8, 4, 2))
        self.assertEqual(first["metrics"]["max_dn"], 70)
        runtime.execute("a-6", "camera_close")
        self.assertEqual(fake.writes(), [1000.0, 5000.0])
        self.assert_shut_down(fake)
        self.assertEqual(runtime.shutdown()["backend_closed"], True)


if __name__ == "__main__":
    unittest.main()
