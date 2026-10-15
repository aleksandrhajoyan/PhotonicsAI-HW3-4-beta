# LUMEN — Autonomous Lensless Measurement & Experimentation Network

**PhotonicsAI Homework 3** · Alexander Hajoyan

LUMEN connects the original HW1 four-agent LangGraph system (Orchestrator, Scientist, General,
Executor) to a real industrial instrument: a lensless iRAYPLE MV-A7200MU130 USB3 monochrome
camera, driven through the manufacturer's official HuaRay IMV SDK. The HW1 graph is reused
unchanged; only its Executor is given a path to the hardware, and that path runs through a small
set of authorized Model Context Protocol (MCP) camera tools.

A language model chooses each physical action — open the camera, set an exposure, take a
measurement, close — one decision at a time, after seeing the measured results of the previous
actions. The model never holds authority: a human must explicitly approve the exact session
envelope before any hardware action, paid API use requires a separate consent, and trusted code
(not the model) enforces every exposure limit, budget, ordering rule and cleanup step.

> **Start here:** the full technical write-up is
> [`submission/HW3_Report_Final.md`](../submission/HW3_Report_Final.md).

---

## Contents

1. [Hardware and software](#1-hardware-and-software)
2. [Architecture](#2-architecture)
3. [Safety architecture](#3-safety-architecture)
4. [Setup and prerequisites](#4-setup-and-prerequisites)
5. [Running the three demonstrations](#5-running-the-three-demonstrations)
6. [Verified physical results](#6-verified-physical-results)
7. [Demonstration videos](#7-demonstration-videos)
8. [Diagrams](#8-diagrams)
9. [Tests and verification](#9-tests-and-verification)
10. [Project documentation](#10-project-documentation)
11. [Limitations](#11-limitations)

---

## 1. Hardware and software

| Component | Details |
|---|---|
| Camera | iRAYPLE MV-A7200MU130, area-scan monochrome, **USB3** |
| Frame format | 1920 × 1200 pixels, **Mono8** (8-bit raw digital numbers, DN) |
| Optics | **No objective lens.** The bare sensor is exposed to ambient room light |
| Operating system | Windows 11 |
| Vendor software | HuaRay **MV Viewer** with the **IMV SDK** (Python binding `IMVApi`, runtime `MVSDKmd.dll`) |
| Language | Python (64-bit; the validated runs used a dedicated virtual environment) |
| Agent framework | LangGraph (`StateGraph`, `interrupt()`, `Command(resume=...)`) |
| Tool protocol | Official MCP Python SDK (server and in-memory `Client`) |
| Language model | OpenAI Responses API (`gpt-6-luna` in the recorded runs), `max_retries=0`, 30 s timeout |
| Numerics | NumPy (in-memory DN statistics only; no frame is written to disk) |

The camera is used purely as a light-integrating sensor. LUMEN does **not** perform optical
focusing or image formation, and the DN values are **not** a calibrated irradiance measurement.
MV Viewer was used only by the operator for pre-checks (device present, `ExposureAuto Off`,
`TriggerMode Off`, original `ExposureTime` and its microsecond unit); it is not part of the LUMEN
control path.

---

## 2. Architecture

### The four roles

LUMEN is implemented by `LumenAgentSystem` ([`lumen_agent_system.py`](lumen_agent_system.py)),
a subclass of the HW1 `AgenticSilverboxSystem` (`task6/agentic_system.py`). The inherited HW1
topology is kept as is:

```
START → orchestrator → scientist | general | executor → orchestrator → … → finish → END
```

| Role | Responsibility | Hardware access |
|---|---|---|
| **Orchestrator** | Decides the next role. Its reply must be exactly `{"next_role", "instruction", "reason"}`; anything else stops the run | None |
| **Scientist** | Hypotheses and research reasoning | None |
| **General** | Analysis of implications and the post-experiment conclusion, grounded in measured DN | None |
| **Executor** | On its first visit, dispatches **once** into the adaptive hardware loop with trusted inputs only; a second visit is refused without any dispatch | **The only role that can reach the camera** |

No text in the agent dialogue grants authority. Hardware approval happens entirely outside the
agent graph.

### Physical execution path

```
Original HW1 four-agent graph (LumenAgentSystem)
  → LUMEN executor bridge            (executor_dialogue_bridge.py)
  → AdaptiveMCPExecutor              (executor_mcp_loop.py; LLM picks one tool call at a time)
  → authorized MCP server            (authorized_mcp.py; five camera tools)
  → SessionGate / AuthorizedCameraRuntime   (session_policy.py, authorized_camera.py)
  → IRaypleCamera                    (irayple_camera.py)
  → official HuaRay IMV SDK
  → real MV-A7200MU130 camera
```

The MCP server runs **in-process** and is reached through the MCP SDK's in-memory client by the
trusted host ([`physical_approved_session_host.py`](physical_approved_session_host.py)). It is
not a network service and is not exposed to any other process or machine.

Key details:

- **Adaptive MCP executor.** `AdaptiveMCPExecutor` lists the server's tools and sends the model
  the research goal, tool schemas, event history, latest measurement and remaining budget. The
  model must answer with exactly one `call_tool` or `finish` action, which is strictly validated
  and executed as-is — never repaired, clipped or retried.
- **Authorized MCP server.** Each tool handler makes exactly one call into
  `AuthorizedCameraRuntime`, with a server-issued `action_id` that a client cannot supply.
- **Camera backend.** `IRaypleCamera` requires exactly one MV-A7200MU130 on USB3, requires
  `TriggerMode`, `ExposureAuto` and `GainAuto` to be `Off`, writes `ExposureTime` with exact
  readback, acquires one Mono8 frame per measurement into NumPy-owned memory, and restores the
  original exposure on close.

See [`docs/system_diagrams.md`](docs/system_diagrams.md) and the architecture and sequence
figures in [Section 8](#8-diagrams).

---

## 3. Safety architecture

Every physical program uses the same trust chain:

1. **Explicit invocation.** A required flag, an explicit `--model` for AI runs (there is no
   default), an interactive terminal, and a typed confirmation phrase (`BEGIN PHYSICAL MCP CHECK`,
   `BEGIN ADAPTIVE HARDWARE CHECK` or `BEGIN FOUR AGENT HARDWARE CHECK`). This confirmation does
   **not** approve hardware access.
2. **Exact human approval of the session envelope (LangGraph HITL).**
   [`hitl_approval.py`](hitl_approval.py) runs a real `START → hardware_approval → END` graph
   that pauses at `interrupt()` and displays the canonical envelope JSON with its **SHA-256**. The
   operator must type exactly `APPROVE <session_id>` within 120 s. Any other input — including an
   extra space, an empty line or a late answer — is a refusal, with no retry.
3. **SessionGate.** Activation requires **both** the approval graph **and** a fresh
   execution-side `SessionGate` ([`session_policy.py`](session_policy.py)) to accept the same
   decision. The gate is a fail-closed state machine; only the `active` state authorizes
   actions, each `action_id` is single-use, and one action may be in flight at a time.
4. **Separate paid-API authorization** (D5A and D6A only). After hardware approval, the operator
   must type `AUTHORIZE PAID API <session_id>` within 60 s; only then is the OpenAI client
   created. Call budgets: D5A at most 7 hardware decisions; D6A at most 16 outer-agent calls plus
   7 hardware decisions. No retries and no fallback model. These are call limits, not a monetary
   spending cap.
5. **Exposure restrictions.** The approved envelope allows 1000–5000 µs. In addition, the trusted
   first-run guard (`FirstRunDecisionGuard` in
   [`physical_adaptive_cli.py`](physical_adaptive_cli.py)) only forwards exposures of exactly
   **1000.0 or 5000.0 µs**.
6. **Tool-call and measurement budgets.** At most 4 tools (`camera_open`, `camera_set_exposure`,
   `camera_measure`, `camera_close`), 2 measurements, 7 tool calls and 180 s per session. Budget
   is reserved before each action and never refunded.
7. **Trusted guard.** The guard never edits a model decision; it forwards it unchanged or refuses
   it, which stops the run before any MCP call. It enforces `camera_open` first, `camera_measure`
   only directly after a new exposure, and only `finish` after `camera_close`.
8. **Shutdown and exposure restoration.** `host.shutdown()` runs on every path — success, refusal,
   validation or tool failure, expiry, `Ctrl+C` and early return. If any exposure write was
   attempted, the original `ExposureTime` is restored and verified before the camera is closed
   and the SDK handle destroyed.
9. **Fail-closed behaviour.** Any failure after authorization is treated as ambiguous: the session
   is aborted and shut down. A failed cleanup is reported as such, never as clean, and the device
   stays claimed until the process is restarted.

> **Warning.** Forcibly terminating the process (for example, killing it from Task Manager or
> closing the terminal window) bypasses the shutdown path and **can prevent restoration of the
> original `ExposureTime`**. If that happens, open MV Viewer and check or restore the exposure
> (5000 µs in the recorded setup).

Exit codes (`physical_adaptive_cli.py`, reused by `physical_lumen_cli.py`; `physical_mcp_smoke.py`
uses the subset that applies without an LLM): `0` PASS · `1` FAIL · `2` usage error or
confirmation refused · `3` not interactive · `4` approval refused · `5` INCOMPLETE · `6` cleanup
failure · `7` API consent refused · `130` interrupted.

---

## 4. Setup and prerequisites

> A complete clean-machine installation has **not** been validated. The steps below describe the
> environment in which the recorded experiments ran and what the source code requires.

**Hardware and vendor software**

- An iRAYPLE MV-A7200MU130 connected over USB3 (exactly one such device must be enumerated).
- HuaRay MV Viewer with its IMV SDK correctly installed. The loader
  ([`irayple_discovery.py`](irayple_discovery.py)) expects the vendor files at their default
  location:
  - `C:\Program Files\HuarayTech\MV Viewer\Development\Samples\Python\IMV\MVSDK\IMVApi.py`
    (and `IMVDefines.py`)
  - `C:\Program Files\HuarayTech\MV Viewer\Runtime\x64\MVSDKmd.dll`
- **MV Viewer must be fully closed** before Python accesses the camera; the device cannot be
  opened by two applications at once.

**Python environment**

- A **64-bit** Python interpreter (the vendor runtime is x64). The recorded runs used Python
  3.12 in a dedicated virtual environment at `%LOCALAPPDATA%\LUMEN\hw3-venv`.
- Python dependencies used by HW3: `langgraph`, `openai`, `numpy`, the official **MCP Python
  SDK** (`mcp==2.3.0`, the tested version; the code uses the v2 API) and `pydantic`, all listed
  in the repository's [`requirements.txt`](../requirements.txt). A fresh install from
  `requirements.txt` on a clean machine has not been validated.

**Credentials (D5A and D6A only)**

- The OpenAI SDK reads `OPENAI_API_KEY` from the process environment when the client is created.
  LUMEN itself never reads, stores, prints or logs credentials.
- Provide the key only to the running process (for example, through a masked prompt in the
  session). Never write it into files, commit it to the repository, or paste it into commands that
  are recorded.
- D4 does not need an API key.

---

## 5. Running the three demonstrations

> **These programs control a physical camera.** Nothing runs automatically: each requires an
> explicit flag, an interactive terminal and typed human approvals. Reading this README or running
> the offline test suite never touches the hardware.

All commands assume the **repository root** as the working directory and `python` being the
configured 64-bit interpreter from [Section 4](#4-setup-and-prerequisites). In PowerShell with the
recorded virtual environment, replace `python` by `& $VenvPython`, where
`$VenvPython = Join-Path $env:LOCALAPPDATA "LUMEN\hw3-venv\Scripts\python.exe"`.

| ID | Command | Model | Paid API | Human gates |
|---|---|---|---|---|
| **D4** — fixed physical MCP diagnostic | `python -B hw3/physical_mcp_smoke.py --physical-mcp-check` | None (fixed six-call script) | **No** | `BEGIN PHYSICAL MCP CHECK`, `APPROVE <session_id>` |
| **D5A** — adaptive LLM | `python -B hw3/physical_adaptive_cli.py --physical-adaptive-check --model gpt-6-luna` | Chooses every MCP call | **Yes** (≤ 7 requests) | `BEGIN ADAPTIVE HARDWARE CHECK`, `APPROVE <session_id>`, `AUTHORIZE PAID API <session_id>` |
| **D6A** — four-agent LUMEN | `python -B hw3/physical_lumen_cli.py --physical-lumen-check --model gpt-6-luna` | Four HW1 roles + hardware decisions | **Yes** (≤ 16 outer + 7 hardware requests) | `BEGIN FOUR AGENT HARDWARE CHECK`, `APPROVE <session_id>`, `AUTHORIZE PAID API <session_id>` |

Important notes:

- **All three** require the real camera, the vendor SDK and human approval at the terminal.
- **D5A and D6A** additionally require API credentials and the explicit paid-API consent.
  `--model` has no default and must always be given.
- **Session IDs are generated freshly for every run** (`lumen-phys-<random hex>`) and printed
  together with the envelope SHA-256. Type the approval lines exactly as printed. Never hardcode,
  script or reuse a session ID; an old ID is refused.
- The fixed D4 sequence is `camera_open → set 1000 µs → measure → set 5000 µs → measure →
  camera_close`. In D5A and D6A the model chooses the sequence within the guard's rules.
- Nothing is saved: no frames, logs or images are written by these programs.

---

## 6. Verified physical results

| Run | Action selection | 1000 µs mean | 5000 µs mean | Saturation | Verdict |
|---|---|---|---|---|---|
| **D4** | Fixed MCP script | 10.9139 DN | 56.6203 DN | 0 / 0 | `PHYSICAL_MCP_CHECK: PASS` |
| **D5A** | LLM, adaptive | 16.0118 DN | 82.0136 DN | 0 / 0 | `PHYSICAL_ADAPTIVE_CHECK: PASS`, exit 0 |
| **D6A** | Four-agent LUMEN, adaptive | 15.5104 DN | 79.4017 DN | 0 / 0 | `PHYSICAL_LUMEN_CHECK: PASS`, exit 0 |

- Each run acquired two full 1920 × 1200 Mono8 hardware frames and verified restoration of the
  original exposure at the end.
- All values are **raw, uncalibrated Mono8 digital numbers** (frame-wide mean), recorded under
  uncontrolled ambient light. They come from the operator's successful terminal reports; the D5A
  and D6A values were also confirmed in the inspected recordings.
- In D5A and D6A, the model chose the 5000 µs exposure **after** measurement 1 (1000 µs) was in
  its input — one feedback-driven exposure correction per run.
- The runs were separate experiments at different times. Absolute DN values must not be compared
  across runs as sensor properties.

Minimum/maximum DN, decision traces and evidence provenance are documented in
[`docs/physical_experiments.md`](docs/physical_experiments.md).

---

## 7. Demonstration videos

The three recordings are **not stored in the Git repository** (`*.mp4` is git-ignored because of
their size). They are planned as assets of a GitHub Release of this repository.

> **Download links pending.** Links will be added here **after** the operator publishes the
> GitHub Release. Until then, only the file names below are given.

| File | Content | Evidence status |
|---|---|---|
| `HW3_01_Physical_MCP_D4.mp4` | D4 fixed physical MCP diagnostic (no model) | Successful run **according to the operator**; not independently re-inspected. It is not claimed to show the exact D4 values in Section 6 |
| `HW3_02_Adaptive_LLM_D5A.mp4` | D5A adaptive LLM experiment (model-selected MCP calls) | Inspected in an earlier review |
| `HW3_03_Four_Agent_LUMEN_D6A.mp4` | D6A four-agent physical LUMEN run (model-selected MCP calls) | Inspected in an earlier review |

Two other local recordings (`demo2 hw3.mp4`, `photonicsai hw 2 Aleqsandr Hajoyan.mp4`) belong to
HW2, are **not** HW3 physical evidence and are not published with HW3.

---

## 8. Diagrams

The Mermaid source of every figure, with a component-to-file trace, is in
[`docs/system_diagrams.md`](docs/system_diagrams.md).

| Figure | Diagram | File |
|---|---|---|
| 1 | HW1 architecture — four-agent LangGraph Silverbox system | [`hw1_architecture.svg`](docs/figures/hw1_architecture.svg) |
| 2 | HW1 sequence — orchestrated dialogue with the coding tool | [`hw1_sequence.svg`](docs/figures/hw1_sequence.svg) |
| 3 | HW2 architecture — HITL code approval and **simulated** focusing | [`hw2_architecture.svg`](docs/figures/hw2_architecture.svg) |
| 4 | HW2 Task 1 sequence — HITL refusal, clarification and approval | [`hw2_hitl_sequence.svg`](docs/figures/hw2_hitl_sequence.svg) |
| 5 | HW2 Task 2 sequence — **simulated** focusing experiment | [`hw2_simulation_sequence.svg`](docs/figures/hw2_simulation_sequence.svg) |
| 6 | **HW3 architecture — LUMEN on the physical iRAYPLE camera** | [`hw3_architecture.svg`](docs/figures/hw3_architecture.svg) |
| 7 | **HW3 sequence — four-agent physical run (D6A)** | [`hw3_sequence.svg`](docs/figures/hw3_sequence.svg) |
| 8 | **Integrated HW1/HW2/HW3 whole-system architecture** (architectural evolution; the three applications run separately) | [`hw1_hw2_hw3_overview.svg`](docs/figures/hw1_hw2_hw3_overview.svg) |

![HW3 LUMEN architecture](docs/figures/hw3_architecture.svg)

---

## 9. Tests and verification

`hw3/tests/` contains 25 offline `unittest` modules. They use a mocked SDK binding, the
synthetic `FakeCamera` and scripted decision providers; they require **no camera, vendor DLL,
network, API key or LLM**, and simulate terminal input where needed.

Run them from the repository root with the configured interpreter:

```powershell
python -B -m unittest discover -s hw3/tests
```

- **1004 passing HW3 offline tests.** The suite was last run on 2026-10-15, after the
  `mcp` and `pydantic` declarations were added to `requirements.txt`: 1004 tests, `OK`, exit
  code 0.
- Leave `OPENAI_API_KEY` unset when running the offline suite.
- The physical experiments (Section 5) are **manual only** and are never part of the automated
  suite.

---

## 10. Project documentation

| Document | Content |
|---|---|
| [`submission/HW3_Report_Final.md`](../submission/HW3_Report_Final.md) | Final HW3 technical report |
| [`hw3/docs/physical_experiments.md`](docs/physical_experiments.md) | Evidence register for D4, D5A and D6A |
| [`hw3/docs/system_diagrams.md`](docs/system_diagrams.md) | HW1–HW3 diagrams (Mermaid source) and traceability |

---

## 11. Limitations

- **No objective lens** is installed; the sensor is bare and forms no optical image.
- **No optical focusing** is demonstrated. (The HW2 focusing results are simulations.)
- **No calibrated irradiance and no linearity claim.** DN values are raw 8-bit numbers with no
  dark-frame, flat-field or gain correction; two exposure points per run cannot establish
  linearity.
- **Ambient illumination is not quantitatively controlled** or measured and may differ between
  and within runs.
- **Only two exposure candidates.** The first-run guard restricts the model to 1000 µs and
  5000 µs, so the runs demonstrate feedback-driven *selection*, not fine exposure optimisation;
  the 70 DN target in the research goal was approached but not reached.
- **One feedback-driven correction per adaptive run.** D5A and D6A each contain exactly one
  exposure correction (1000 µs → 5000 µs); no session performs multiple sequential corrections.
- **Platform dependency.** Hardware execution depends on Windows, the HuaRay IMV SDK at its
  default install path, a 64-bit Python and the specific camera model. Only the offline tests are
  hardware-independent.
- **Crash limitation.** A forcibly terminated process cannot restore the original `ExposureTime`.
