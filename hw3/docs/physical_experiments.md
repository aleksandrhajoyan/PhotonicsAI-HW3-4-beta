# HW3 Physical Experiment Evidence Register (D4, D5A, D6A)

Register date: 2026-10-15. This document records three successful experiments on the physical
iRAYPLE MV-A7200MU130 lensless monochrome camera. Writing it involved no camera, SDK, API or
network access, and no existing file or media file was changed.

## Evidence sources and how to read this register

| Source label | Meaning |
|---|---|
| **TERMINAL REPORT** | The operator's verbatim terminal results for the run, supplied to the implementer. This is the primary numerical record |
| **VIDEO-VERIFIED** | The value or event was independently read from frames of the recording during this audit |
| **SOURCE** | The behaviour is enforced or printed by the named script (read during this audit) |
| **NOT IN VIDEO** | The item is in the terminal report but was not visible in any inspected frame |

How the videos were inspected:

- No video metadata tools (ffprobe, ffmpeg, mediainfo, exiftool) are installed.
- Durations come from Windows Shell metadata.
- Frames were decoded by the locally installed Chrome. Each video was sampled every 4–6 s, with
  1 s or finer steps around the final summaries. Frames were written to a temporary directory
  outside the repository.
- The SHA-256 of every inspected video was identical before and after inspection.

No run log, image or raw frame from these experiments exists in the repository. By design, the
CLIs save nothing.

---

## 1. Experiment overview and purposes

| ID | Script | Purpose |
|---|---|---|
| **D4** | `hw3/physical_mcp_smoke.py --physical-mcp-check` | Verify that the authorized MCP server can operate the physical camera end to end through `SessionGate` and `IRaypleCamera`, using a **fixed, transparent six-call sequence**. No AI decides anything |
| **D5A** | `hw3/physical_adaptive_cli.py --physical-adaptive-check --model gpt-6-luna` | Verify that a **language model selects every physical MCP action**, one decision at a time, after seeing the earlier results, and that it uses measurement 1 to choose the second exposure |
| **D6A** | `hw3/physical_lumen_cli.py --physical-lumen-check --model gpt-6-luna` | Verify that the **original HW1 four-role LangGraph system** (Orchestrator, Scientist, General, Executor) can direct a physical experiment. The Executor dispatches once to the D5A adaptive MCP executor, and a post-experiment conclusion is required |

All three runs share the same human-approved session envelope, built by
`physical_mcp_smoke.build_envelope`:

- 4 tools: `camera_open`, `camera_set_exposure`, `camera_measure`, `camera_close`
- exposure 1000–5000 µs
- at most 2 measurements, 7 tool calls and 180 s

---

## 2. Experiment comparison

| Item | D4 | D5A | D6A |
|---|---|---|---|
| Who chooses the MCP calls | Fixed script | LLM (`gpt-6-luna`) | LLM (`gpt-6-luna`) inside the four-agent graph |
| AI-driven | **No** | Yes | Yes |
| HW1 four-role graph | No | No | **Yes** |
| Paid OpenAI API requests | 0 | 7 | 15 (8 outer-agent + 7 hardware Executor) |
| Authorized MCP tool calls | 6 | 6 | 6 |
| Measurements (1920×1200 Mono8, hardware) | 2 | 2 | 2 |
| 1000 µs: mean / min / max DN | 10.9139 / 6 / 14 | 16.0118 / 11 / 20 | 15.5104 / 11 / 18 |
| 5000 µs: mean / min / max DN | 56.6203 / 32 / 70 | 82.0136 / 59 / 96 | 79.4017 / 63 / 89 |
| Saturation fraction | 0 / 0 | 0 / 0 | 0 / 0 |
| Feedback-driven exposure choice | No (predetermined) | Yes, one (decision 4) | Yes, one (decision 4) |
| Exposure restoration verified | Yes | Yes | Yes |
| Verdict | `PHYSICAL_MCP_CHECK: PASS` | `PHYSICAL_ADAPTIVE_CHECK: PASS`, exit 0 | `PHYSICAL_LUMEN_CHECK: PASS`, exit 0 |
| Video recording | `HW3_01_Physical_MCP_D4.mp4` (operator-confirmed; not inspected) | `HW3_02_Adaptive_LLM_D5A.mp4` (inspected) | `HW3_03_Four_Agent_LUMEN_D6A.mp4` (inspected) |

All values in this table come from the terminal reports. Section 6 shows which of them were also
verified in video.

---

## 3. D4: fixed physical MCP diagnostic

**Source of record:** TERMINAL REPORT. The recording `HW3_01_Physical_MCP_D4.mp4` is a successful
D4 run **according to the operator**, but it was not independently inspected. It may show a
different successful D4 run, so it is not claimed to contain the values below (see Section 6).

**What happened.** After the typed confirmation `BEGIN PHYSICAL MCP CHECK` and the human approval
`APPROVE <session_id>` of the exact session envelope (HITL), the script issued exactly six
authorized MCP calls:

1. `camera_open`
2. `camera_set_exposure(1000.0)`
3. `camera_measure`
4. `camera_set_exposure(5000.0)`
5. `camera_measure`
6. `camera_close`

The sequence is fixed in the source (SOURCE: `physical_mcp_smoke.py` docstring, *"This is NOT an
LLM-driven experiment"*).

| Measurement | Exposure | Format | Mean DN | Min DN | Max DN | Saturation fraction |
|---|---|---|---|---|---|---|
| 1 | 1000 µs | 1920×1200 Mono8 | 10.9139 | 6 | 14 | 0 |
| 2 | 5000 µs | 1920×1200 Mono8 | 56.6203 | 32 | 70 | 0 |

- Mean difference (measurement 2 − measurement 1): **+45.7064 DN**
- Exposure restoration: verified
- Verdict: `PHYSICAL_MCP_CHECK: PASS`. The verdict string is printed by `physical_mcp_smoke.py:534`.

**Conclusion supported:** direct, gated physical MCP operation works on real hardware.
**Not supported:** any claim of LLM control. No model was involved.

---

## 4. D5A: real LLM adaptive experiment

**Sources of record:** TERMINAL REPORT, with most values also VIDEO-VERIFIED in
`HW3_02_Adaptive_LLM_D5A.mp4` (inspected under its original name, `demo3 hw3 adaptive llm.mp4`).

**Run flow:**

1. **Launch (video ~9 s):**
   `& $VenvPython -B hw3\physical_adaptive_cli.py --physical-adaptive-check --model gpt-6-luna`.
   The banner reads `LUMEN physical adaptive check (HW3.7-D5A)`.
2. **Three human gates (video ~9–37 s):**
   - `BEGIN ADAPTIVE HARDWARE CHECK`
   - `APPROVE lumen-phys-fc6e0919ae5180e1864a56e805f2d54b`, after the session approval request
     was shown with the envelope SHA-256 `bb184dbfcbd85732…`
   - `AUTHORIZE PAID API lumen-phys-fc6e0919ae5180e1864a56e805f2d54b`, after a paid-API notice of
     "at most 7" requests
3. **Model decisions (VIDEO-VERIFIED):** the model made 7 decisions, each `check: passed` by the
   trusted `FirstRunDecisionGuard`. Decision 7 was `finish`.
4. **MCP calls (VIDEO-VERIFIED):** 6 calls, all `success`. The applied exposures read back as
   1000.0 µs and 5000.0 µs.

| Measurement | Exposure | Format | Mean DN | Min DN | Max DN | Saturation fraction | Video |
|---|---|---|---|---|---|---|---|
| 1 (frame_id 1, source hardware) | 1000 µs | 1920×1200 Mono8 | 16.0118 | 11 | 20 | 0.000000 | VIDEO-VERIFIED |
| 2 (frame_id 2, source hardware) | 5000 µs | 1920×1200 Mono8 | 82.0136 | 59 | 96 | 0.000000 | VIDEO-VERIFIED |

**Feedback evidence (VIDEO-VERIFIED):** *"decision 4 (event 7) was requested after measurement
frame_id 1 (event 6, mean 16.0118 DN) was in its input and chose camera_set_exposure(5000.0);
measurement frame_id 2 followed (event 10)"*.

**Cleanup and verdict (VIDEO-VERIFIED):**

- `session state: closed before trusted shutdown`
- `host shutdown: OK … ExposureTime restored and verified`
- `device ownership: released`
- `PHYSICAL_ADAPTIVE_CHECK: PASS`
- `exit code: 0`

**Paid API requests:** 7 per the TERMINAL REPORT. In the video, **6** `HTTP/1.1 200 OK` request
lines are visible at ~45 s. The summary had replaced them on screen by ~46 s, so a 7th request line
was not captured in any sampled frame. The 7 listed model decisions are consistent with 7 requests.

---

## 5. D6A: four-agent physical LUMEN

**Sources of record:** TERMINAL REPORT, with most values also VIDEO-VERIFIED in
`HW3_03_Four_Agent_LUMEN_D6A.mp4` (inspected under its original name, `demo1 hw3.mp4`).

**Run flow:**

1. **Launch (video ~7 s):**
   `& $VenvPython -B hw3\physical_lumen_cli.py --physical-lumen-check --model gpt-6-luna`.
   The banner reads `LUMEN four-agent physical check (HW3.7-D6A)`.
2. **Three human gates (video ~7–43 s):**
   - `BEGIN FOUR AGENT HARDWARE CHECK`
   - `APPROVE lumen-phys-05d1fcac1270faecf49f4a4c0212a6f8`, after the envelope SHA-256
     `2a86b0eb9a372c7f…` was shown
   - `AUTHORIZE PAID API lumen-phys-05d1fcac1270faecf49f4a4c0212a6f8`

**Agents (TERMINAL REPORT; NOT IN VIDEO).** All four original HW1 roles took part, in this order:

orchestrator → scientist → orchestrator → general → orchestrator → executor → orchestrator →
general → orchestrator

- **Video:** the role list scrolled off-screen before it could be seen. The video shows only the
  verdict text *"four roles took part"*.
- **Source:** the PASS verdict requires a Scientist or General conclusion after the measurements
  (`physical_lumen_cli.py:221`). The TERMINAL REPORT states that General wrote it.

**API requests:**

- 8 outer-agent + 7 hardware Executor requests (TERMINAL REPORT)
- 15 `HTTP/1.1 200 OK` lines are visible in the video (VIDEO-VERIFIED). The 8/7 split is NOT IN
  VIDEO.

**Hardware actions (VIDEO-VERIFIED):** 7 hardware model decisions, all `check: passed`, and 6 MCP
calls, all `success`.

| Measurement | Exposure | Format | Mean DN | Min DN | Max DN | Saturation fraction | Video |
|---|---|---|---|---|---|---|---|
| 1 (frame_id 1, source hardware) | 1000 µs | 1920×1200 Mono8 | 15.5104 | 11 | 18 | 0.000000 | VIDEO-VERIFIED |
| 2 (frame_id 2, source hardware) | 5000 µs | 1920×1200 Mono8 | 79.4017 | 63 | 89 | 0.000000 | VIDEO-VERIFIED |

**Feedback evidence (VIDEO-VERIFIED):** decision 4 (event 7) was requested after measurement
frame_id 1 (event 6, mean 15.5104 DN) and chose `camera_set_exposure(5000.0)`.

**Cleanup and verdict (VIDEO-VERIFIED):**

- session closed before trusted shutdown
- ExposureTime restored and verified
- `device ownership: released`
- `PHYSICAL_ADAPTIVE_CHECK: PASS`, then `PHYSICAL_LUMEN_CHECK: PASS (four roles took part; two
  hardware measurements with a feedback-driven second exposure; verified shutdown)`
- `exit code: 0`

---

## 6. Video evidence

The operator renamed the physical recordings to the names below. The D5A and D6A files were
inspected under their original names. In HW3.10-A, their SHA-256 values were confirmed identical
under the new names, so the content is the same.

| Relative path | Size (bytes) | Duration | Content | Status |
|---|---|---|---|---|
| `HW3_01_Physical_MCP_D4.mp4` | not recorded here | not recorded here | Successful **D4** fixed physical MCP run, **according to the operator**. It replaces an earlier recording | **Operator-confirmed; not independently inspected** |
| `HW3_02_Adaptive_LLM_D5A.mp4` (formerly `demo3 hw3 adaptive llm.mp4`) | 56,337,631 | 54 s (54.7 s) | **D5A** LLM adaptive physical run (banner `HW3.7-D5A`, session `lumen-phys-fc6e0919…`) | **VIDEO-VERIFIED** |
| `HW3_03_Four_Agent_LUMEN_D6A.mp4` (formerly `demo1 hw3.mp4`) | 95,763,388 | 1 min 35 s (95.8 s) | **D6A** four-agent physical LUMEN run (banner `HW3.7-D6A`, session `lumen-phys-05d1fcac…`) | **VIDEO-VERIFIED** |
| `demo2 hw3.mp4` | 49,846,826 | 41 s (41.1 s) | **Not a physical experiment.** It shows the HW2 Task 2 *simulated* focusing live demo (`python -m hw2.demo_task2_live --model gpt-6-luna`, run directory `hw2_task2_live_20261012_211839`, "Simulation only … simulated bench"). Despite its filename, it contains no HW3 hardware run | Inspected, **not HW3 evidence** |
| `photonicsai hw 2 Aleqsandr Hajoyan.mp4` | 62,259,513 | 3 min 30 s | HW2 submission video (by name). Not content-inspected | Not HW3 evidence |

**D4 recording.** The D4 values in Section 3 come from the operator's terminal report. The current
`HW3_01_Physical_MCP_D4.mp4` is a successful D4 recording according to the operator, but it was
not independently inspected. It is therefore not claimed to contain those exact values.

Additional notes on the inspected videos:

- The D6A recording also shows, in its first ~7 s, leftover scrollback from an **earlier**
  D5A-type run: mean 16.1011 → 83.2714 DN, `PHYSICAL_ADAPTIVE_CHECK: PASS`, exit 0. That earlier
  run is not the D5A run registered here.
- Both inspected physical videos show the terminal only. The camera and MV Viewer are not on
  screen.
- In the D5A recording, the operator enters the OpenAI key through a masked
  `Read-Host -AsSecureString` prompt. The key is not visible.

Video SHA-256 values of the inspected files (unchanged by inspection, and identical under the new
names):

| File | SHA-256 |
|---|---|
| `HW3_03_Four_Agent_LUMEN_D6A.mp4` | `4E9C4C1E0912C7C26A13D091FF495F2CC21CA48E1C2926158103B5ADBB0288BF` |
| `HW3_02_Adaptive_LLM_D5A.mp4` | `F705C0A9E5F9136E239DE74EEA347CED2C7761754CF9513E5EE657AD389FEAC1` |
| `demo2 hw3.mp4` | `04C25EF1FFEA9DAF1BB4F10916A2C3C517C986C7E409EBAF15AE21E94F206020` |

---

## 7. Safety, HITL and cleanup evidence

| Control | D4 | D5A | D6A | Evidence |
|---|---|---|---|---|
| Explicit flag and interactive terminal required | Yes | Yes | Yes | SOURCE (`_parser()`, `_is_interactive`) |
| Typed first confirmation, which does not approve hardware | `BEGIN PHYSICAL MCP CHECK` | `BEGIN ADAPTIVE HARDWARE CHECK` | `BEGIN FOUR AGENT HARDWARE CHECK` | SOURCE. Video for D5A and D6A |
| LangGraph session approval `APPROVE <session_id>` of the exact SHA-256-bound envelope | Yes | Yes | Yes | Video for D5A and D6A. TERMINAL REPORT for D4 |
| Separate paid-API consent `AUTHORIZE PAID API <session_id>` | n/a (no API) | Yes | Yes | Video |
| Trusted first-run guard: exposure only 1000.0 or 5000.0 µs; order open → (set → measure) ≤ 2 → close | n/a (fixed script) | Yes, every decision `check: passed` | Yes, every decision `check: passed` | Video and SOURCE |
| Envelope limits (≤ 2 measurements, ≤ 7 tool calls, 180 s) | Yes | Yes (6 calls) | Yes (6 calls) | Video and SOURCE |
| Exposure restored and verified at close | Yes | Yes | Yes | TERMINAL REPORT. Video for D5A and D6A |
| Shutdown and device ownership released | Yes | Yes | Yes | TERMINAL REPORT. Video for D5A and D6A |
| No secrets displayed | n/a | Yes (masked key entry) | Yes | Video |

---

## 8. Scientific interpretation and explicit limits

**What the three experiments show:**

- **D4** verifies direct physical MCP operation: authorized server → `SessionGate` →
  `IRaypleCamera` → real Mono8 frames, with verified restoration. It does **not** demonstrate LLM
  control.
- **D5A** verifies **LLM-selected physical actions with feedback**. Every MCP call was chosen by
  `gpt-6-luna`, and the second exposure was chosen after measurement 1 was in the model's input.
- **D6A** verifies **integration of the original HW1 four-agent system** with physical MCP
  execution. The four roles coordinate, the Executor runs the adaptive hardware loop once, and a
  post-experiment conclusion based on the measured DN is written.
- **D5A and D6A are two independent adaptive experiments**, each in its own approved session.
- **Each adaptive experiment contains exactly one feedback-driven exposure correction**
  (1000 µs → 5000 µs, decision 4). No session contains two sequential exposure corrections. The
  policy allows at most two measurements per session.

**Limits of the conclusions:**

- All values are **raw, uncalibrated digital numbers (DN)** from 8-bit Mono8 frames. No dark-frame
  correction, flat-field or gain calibration was applied.
- **Illumination was ambient and was not quantitatively controlled, measured or calibrated.** It
  may have differed between and within the runs.
  - This is the likely reason the absolute DN values differ between the experiments (for example,
    10.9 vs 16.0 vs 15.5 DN at 1000 µs).
  - Comparisons across experiments are therefore not meaningful as sensor properties.
- **No linearity is demonstrated.** Two exposure points per run are not a linearity test.
- **No irradiance measurement, no sensor calibration, no sensitivity or noise characterisation.**
- **No focus optimisation or image formation.** The sensor is lensless.
- **The model's choice was restricted** to two candidate exposures (1000 or 5000 µs) by the trusted
  guard. The runs demonstrate feedback-driven selection, not fine exposure optimisation.
  - The 5000 µs means (56.6, 82.0 and 79.4 DN) are near, but not at, the 70 DN target in the task
    prompt.
- **D4 values come from the TERMINAL REPORT.** The D4 recording is operator-confirmed but was not
  independently inspected.
- **Some D6A details appear only in the TERMINAL REPORT:** the role order, the 8/7 API split and
  the General conclusion text. The video shows only the verdict line.
- **The 7th D5A API request line was not seen in the sampled frames.**

---

## 9. Traceability

| Item | Source file(s) | Diagram |
|---|---|---|
| D4 fixed six-call sequence, `build_envelope`, `PHYSICAL_MCP_CHECK` verdict | `hw3/physical_mcp_smoke.py` | Figures 6, 7 (envelope values) |
| D5A adaptive CLI, `FirstRunDecisionGuard`, `EXPOSURE_CANDIDATES_US`, feedback evidence | `hw3/physical_adaptive_cli.py` | Figures 6, 7 |
| D6A CLI, three gates, `OuterCallBudget`, verdict including the conclusion requirement | `hw3/physical_lumen_cli.py` | Figures 6, 7 |
| Four-role graph reused from HW1 | `hw3/lumen_agent_system.py` (`LumenAgentSystem(AgenticSilverboxSystem)`), `task6/agentic_system.py` | Figures 1, 6 |
| Adaptive MCP loop and bridge | `hw3/executor_mcp_loop.py`, `hw3/executor_dialogue_bridge.py`, `hw3/llm_decision_provider.py`, `hw3/openai_model_caller.py` | Figures 6, 7 |
| Approval graph, host, gate | `hw3/hitl_approval.py`, `hw3/approved_session_host.py`, `hw3/physical_approved_session_host.py`, `hw3/session_policy.py` | Figures 6, 7 |
| Authorized MCP server and runtime | `hw3/authorized_mcp.py`, `hw3/authorized_camera.py` | Figures 6, 7 |
| Camera backend (HuaRay IMV SDK) and DN metrics | `hw3/irayple_camera.py`, `hw3/measurements.py` | Figures 6, 7 |

Diagram files:

- Source: `hw3/docs/system_diagrams.md` (Figure 6 = HW3 architecture, Figure 7 = HW3 sequence,
  which follows the D6A run)
- Rendered: `hw3/docs/figures/hw3_architecture.svg`, `hw3/docs/figures/hw3_sequence.svg`
- HW1 lineage: `hw3/docs/figures/hw1_architecture.svg`

Related audit: `hw3/docs/submission_readiness.md`. That audit was written before the D5A recording
existed, so its video inventory is superseded by Section 6 here.
