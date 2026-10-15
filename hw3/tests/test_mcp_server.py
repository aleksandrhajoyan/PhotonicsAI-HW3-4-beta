import ast
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters

HW3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW3_DIR))

from fake_camera import FakeCamera  # noqa: E402
from mcp_server import create_server  # noqa: E402
from measurements import analyze_frame  # noqa: E402

SERVER_PATH = HW3_DIR / "mcp_server.py"
EXPECTED_TOOLS = {
    "camera_info",
    "camera_open",
    "camera_set_exposure",
    "camera_measure",
    "camera_close",
}


def _error_text(result):
    return " ".join(getattr(block, "text", "") for block in result.content)


class McpServerTests(unittest.IsolatedAsyncioTestCase):
    """Protocol-level tests: every call goes through the official MCP Client."""

    def setUp(self):
        # Fresh backend and server per test to prevent cross-test contamination.
        self.camera = FakeCamera()
        self.server = create_server(self.camera)

    @asynccontextmanager
    async def connect(self):
        async with Client(self.server, mode="legacy", read_timeout_seconds=30) as client:
            yield client

    async def call_ok(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertFalse(result.is_error, f"{name} failed: {_error_text(result)}")
        self.assertIsInstance(result.structured_content, dict)
        return result.structured_content

    async def call_error(self, client, name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        self.assertTrue(result.is_error, f"{name} unexpectedly succeeded")
        self.assertIsNone(result.structured_content)
        return _error_text(result)

    async def measure_at(self, client, exposure_us):
        await self.call_ok(client, "camera_set_exposure", {"exposure_us": exposure_us})
        return await self.call_ok(client, "camera_measure")

    async def test_server_initializes(self):
        async with self.connect() as client:
            self.assertEqual(client.server_info.name, "lumen-hw3-fake-camera")
            self.assertIsNotNone(client.server_capabilities.tools)
            self.assertIn("synthetic", client.instructions)

    async def test_list_tools_returns_exactly_five(self):
        async with self.connect() as client:
            tools = (await client.list_tools()).tools
        names = [tool.name for tool in tools]
        self.assertEqual(len(names), 5)
        self.assertEqual(set(names), EXPECTED_TOOLS)
        for tool in tools:
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description)
        set_exposure = next(t for t in tools if t.name == "camera_set_exposure")
        self.assertEqual(set_exposure.input_schema["required"], ["exposure_us"])
        self.assertEqual(
            set_exposure.input_schema["properties"]["exposure_us"]["type"], "number"
        )

    async def test_camera_info_is_synthetic_and_works_closed(self):
        async with self.connect() as client:
            info = await self.call_ok(client, "camera_info")
        self.assertEqual(info["source"], "synthetic")
        self.assertIs(info["synthetic"], True)
        self.assertEqual(info["backend"]["backend"], "FakeCamera")
        self.assertIs(info["backend"]["is_open"], False)
        self.assertEqual(info["exposure_limits_us"], {"min": 100.0, "max": 10000.0})

    async def test_measure_before_open_rejected(self):
        async with self.connect() as client:
            text = await self.call_error(client, "camera_measure")
        self.assertIn("open", text)
        self.assertFalse(self.camera.is_open)

    async def test_set_exposure_before_open_rejected(self):
        async with self.connect() as client:
            text = await self.call_error(client, "camera_set_exposure", {"exposure_us": 2000.0})
        self.assertIn("open", text)
        self.assertEqual(self.camera.get_info()["exposure_us"], 1000.0)

    async def test_open_succeeds_and_is_idempotent(self):
        async with self.connect() as client:
            first = await self.call_ok(client, "camera_open")
            second = await self.call_ok(client, "camera_open")
            info = await self.call_ok(client, "camera_info")
        self.assertIs(first["ok"], True)
        self.assertEqual(first, second)
        self.assertEqual(first["source"], "synthetic")
        self.assertTrue(self.camera.is_open)
        self.assertIs(info["backend"]["is_open"], True)

    async def test_valid_exposure_change_succeeds(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.call_ok(client, "camera_set_exposure", {"exposure_us": 2500.5})
            info = await self.call_ok(client, "camera_info")
        self.assertEqual(result["exposure_us"], 2500.5)
        self.assertEqual(info["backend"]["exposure_us"], 2500.5)

    async def test_invalid_exposure_rejected(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            for value in (99.999, 0, -1.0, 10000.001, 1e9, "1000", True, None):
                with self.subTest(value=value):
                    await self.call_error(client, "camera_set_exposure", {"exposure_us": value})
            await self.call_error(client, "camera_set_exposure", {})

    async def test_failed_exposure_change_keeps_state(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_set_exposure", {"exposure_us": 2000.0})
            await self.call_error(client, "camera_set_exposure", {"exposure_us": 20000.0})
            info = await self.call_ok(client, "camera_info")
            measurement = await self.call_ok(client, "camera_measure")
        self.assertEqual(info["backend"]["exposure_us"], 2000.0)
        self.assertEqual(measurement["exposure_us"], 2000.0)

    async def test_measure_returns_synthetic_metadata(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.call_ok(client, "camera_measure")
        self.assertEqual(result["source"], "synthetic")
        self.assertIs(result["synthetic"], True)
        self.assertEqual(result["frame_id"], 1)
        self.assertEqual(result["exposure_us"], 1000.0)
        self.assertEqual(result["bit_depth"], 8)
        self.assertEqual((result["width"], result["height"]), (8, 8))
        self.assertIn("synthetic", result["note"].lower())
        self.assertNotIn("pixels", result)
        self.assertEqual(result["metrics"]["histogram"]["bins"], 16)
        self.assertEqual(len(result["metrics"]["histogram"]["counts"]), 16)

    async def test_measure_metrics_agree_with_analyze_frame(self):
        reference_camera = FakeCamera()
        reference_camera.open()
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            for exposure in (1000.0, 2000.0):
                with self.subTest(exposure=exposure):
                    result = await self.measure_at(client, exposure)
                    reference_camera.set_exposure_us(exposure)
                    frame = reference_camera.capture_frame()
                    expected = analyze_frame(frame.pixels, frame.bit_depth, bins=16)
                    self.assertEqual(result["metrics"], expected)
        self.assertEqual(expected["saturation_fraction"], 15 / 64)

    async def test_higher_exposure_increases_brightness(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            means = []
            for exposure in (100.0, 500.0, 1000.0, 1500.0):
                metrics = (await self.measure_at(client, exposure))["metrics"]
                self.assertEqual(metrics["saturation_fraction"], 0.0)
                means.append(metrics["mean_dn"])
        self.assertEqual(means, sorted(means))
        self.assertEqual(len(set(means)), len(means))

    async def test_max_exposure_saturates(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            result = await self.measure_at(client, 10000.0)
        self.assertEqual(result["exposure_us"], 10000.0)
        self.assertEqual(result["metrics"]["saturation_fraction"], 1.0)
        self.assertEqual(result["metrics"]["mean_dn"], 255.0)
        self.assertEqual(result["metrics"]["min_dn"], 255)

    async def test_frame_ids_increase(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            ids = [(await self.call_ok(client, "camera_measure"))["frame_id"] for _ in range(3)]
        self.assertEqual(ids, [1, 2, 3])

    async def test_close_succeeds_and_is_safe_when_closed(self):
        async with self.connect() as client:
            already_closed = await self.call_ok(client, "camera_close")
            await self.call_ok(client, "camera_open")
            closed = await self.call_ok(client, "camera_close")
            info = await self.call_ok(client, "camera_info")
        self.assertIs(already_closed["ok"], True)
        self.assertIs(closed["ok"], True)
        self.assertIs(closed["is_open"], False)
        self.assertFalse(self.camera.is_open)
        self.assertIs(info["backend"]["is_open"], False)

    async def test_measure_after_close_rejected(self):
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
            await self.call_ok(client, "camera_measure")
            await self.call_ok(client, "camera_close")
            text = await self.call_error(client, "camera_measure")
        self.assertIn("open", text)

    async def test_isolated_servers_do_not_share_state(self):
        other_camera = FakeCamera()
        other_server = create_server(other_camera)
        async with self.connect() as client:
            await self.call_ok(client, "camera_open")
        async with Client(other_server, mode="legacy") as other_client:
            await self.call_error(other_client, "camera_measure")
        self.assertTrue(self.camera.is_open)
        self.assertFalse(other_camera.is_open)


class FactoryTests(unittest.TestCase):
    def test_rejects_non_synthetic_backend(self):
        class HardwareLabelled(FakeCamera):
            def get_info(self):
                return {**super().get_info(), "source": "hardware", "synthetic": False}

        with self.assertRaises(ValueError):
            create_server(HardwareLabelled())

    def test_rejects_non_backend(self):
        with self.assertRaises(TypeError):
            create_server(object())


class NoVendorSdkTests(unittest.TestCase):
    ALLOWED_IMPORTS = {
        "typing",
        "mcp",
        "pydantic",
        "camera_contract",
        "fake_camera",
        "measurements",
    }

    def test_server_imports_only_allowed_modules(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertTrue(
            imported <= self.ALLOWED_IMPORTS,
            f"unexpected imports: {imported - self.ALLOWED_IMPORTS}",
        )
        self.assertNotIn("ctypes", imported)
        self.assertNotIn("sdk_probe", imported)

    def test_server_source_has_no_vendor_references(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        for token in ("MVSDK", "sdk_probe", "ctypes", "iRAYPLE", ".dll"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_sdk_probe_not_loaded(self):
        self.assertNotIn("sdk_probe", sys.modules)


class StdioSubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_client_calls_camera_info(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-B", str(SERVER_PATH)],
            cwd=str(HW3_DIR.parent),
        )
        async with Client(params, read_timeout_seconds=60) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool("camera_info", {})
        self.assertEqual(names, EXPECTED_TOOLS)
        self.assertFalse(result.is_error, _error_text(result))
        info = result.structured_content
        self.assertEqual(info["source"], "synthetic")
        self.assertIs(info["synthetic"], True)
        self.assertEqual(info["backend"]["backend"], "FakeCamera")


if __name__ == "__main__":
    unittest.main()
