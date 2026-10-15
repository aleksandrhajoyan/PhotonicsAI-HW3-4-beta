# LUMEN / PhotonicsAI — System Architecture and Sequence Diagrams (HW1, HW2, HW3)

This document gives editable Mermaid diagrams of the three homework systems. Every diagram was
drawn from the source code in this repository. Section 5 maps each diagram component to its
source file.

## 1. Architecture evolution

All three homeworks use **the same four-role LangGraph system** from HW1
(`task6/agentic_system.py`, class `AgenticSilverboxSystem`). Later homeworks subclass it. They do
not modify it.

| Homework | What the four roles control | Executor's only capability | Human role | Physical hardware |
|---|---|---|---|---|
| **HW1** (Tasks 5/6) | Silverbox nonlinear-oscillator modelling (neural acceleration network + RK4 free run, the "PINN / neural state-space" task) | `PythonCodingTool`: writes a new Python script and runs it as a subprocess (300 s timeout) | None inside the graph | **None.** The data is a recorded benchmark file (`SNLS80mV.mat`) |
| **HW2 Task 1** | The same Silverbox work, plus a model-selected `human` destination | The same coding tool. It runs only after the human approves the exact SHA-256-bound action | LangGraph `interrupt()` / `Command(resume=...)` for clarification, code approval and final-measurement approval | **None** |
| **HW2 Task 2** | Simulated wavefront shaping (focusing through a scattering medium) | `FocusingSession.call(arguments)`, one bounded simulated experiment | None (the simulated tool needs no approval) | **None.** DMD, scattering medium and camera are all **simulated** |
| **HW3** (LUMEN) | Camera-exposure experiment on a real sensor | The Executor dispatches **once**, through a bridge, to an adaptive MCP executor. Each MCP tool call is chosen by an LLM and enforced by `SessionGate` | Typed first confirmation, real LangGraph session approval, separate paid-API consent | **Yes.** iRAYPLE MV-A7200MU130, a **lensless monochrome** sensor, through the official HuaRay IMV SDK |

Key distinctions:

- HW1's PINN/Silverbox system is a **numerical modelling task on recorded data**. It is distinct
  from HW3's physical camera and shares no hardware path with it.
- HW2's optics experiment is **simulated**. The DMD, the transmission matrix *T* and the camera
  noise are generated in software (`hw2/simulated_bench.py`). No hardware was used.
- HW3 uses a **real lensless monochrome sensor**. Its data are **uncalibrated digital numbers
  (DN)** from Mono8 frames.

---

## 2. HW1 — Four-agent LangGraph system (Tasks 5 and 6)

### Figure 1 — HW1 architecture

```mermaid
flowchart TB
    subgraph T5["Task 5: hand-written direct solution, not produced by the agents"]
        T5S["task5/task5_silverbox_pinn.py<br/>PyTorch acceleration network + RK4 free run"]
        T5R["task5/task5_results.json<br/>recorded train/test free-run RMSE"]
        T5S --> T5R
    end

    ASSIGN["task6/task5_assignment.txt<br/>Task 5 prompt used as the user goal"]
    DATA[("SilverboxFiles/SNLS80mV.mat<br/>recorded benchmark data")]

    subgraph T6["Task 6: AgenticSilverboxSystem in task6/agentic_system.py"]
        MAIN["main()<br/>checks OPENAI_API_KEY is set, locates data,<br/>creates task6_runs/run_timestamp"]
        subgraph LG["LangGraph StateGraph of DialogueState<br/>shared dialogue of role/text turns, appended with operator.add"]
            START1(["START"])
            ORC["orchestrator<br/>LLM routing JSON: next_role, instruction, reason<br/>up to 3 attempts with rejection feedback<br/>finish gate + quality gate TEST_RMSE below 0.02 V"]
            SCI["scientist<br/>LLM: Duffing equation, network, RK4,<br/>free-run protocol, numbered plan<br/>no tools"]
            GEN["general<br/>LLM: side questions, terminology, metrics<br/>no tools"]
            EXE["executor<br/>LLM writes one complete Python script<br/>the only role with the coding tool"]
            END1(["END"])
        end
        MC["ModelCaller<br/>OpenAI Responses API<br/>default model gpt-6-luna"]
        TOOL["PythonCodingTool.write_and_run<br/>new attempt directory and new file,<br/>subprocess with 300 s hard timeout"]
        SCRIPT["Generated attempt script<br/>trains network, one RK4 free run over 4096 samples,<br/>prints TRAIN_RMSE and TEST_RMSE, saves artifacts"]
        EVID["Run evidence<br/>dialogue.json, dialogue.txt, run_summary.json"]
    end

    VER["task6/verify_run.py<br/>independent verifier, checks A to D"]

    ASSIGN --> MAIN
    MAIN -->|"graph.stream"| START1
    START1 --> ORC
    ORC -->|"scientist"| SCI
    ORC -->|"general"| GEN
    ORC -->|"executor"| EXE
    ORC -->|"finish: needs exit_code=0 and a printed TEST_RMSE"| END1
    SCI -->|"fewer than 8 decisions"| ORC
    GEN -->|"fewer than 8 decisions"| ORC
    EXE -->|"fewer than 8 decisions"| ORC
    SCI -.->|"8th decision used"| END1
    GEN -.->|"8th decision used"| END1
    EXE -.->|"8th decision used"| END1
    ORC -.-> MC
    SCI -.-> MC
    GEN -.-> MC
    EXE -.-> MC
    EXE --> TOOL
    TOOL -->|"runs"| SCRIPT
    SCRIPT -->|"reads via SILVERBOX_MAT_PATH"| DATA
    SCRIPT -->|"exit code, stdout, stderr, traceback"| TOOL
    TOOL -->|"executor turn appended to dialogue"| EXE
    MAIN --> EVID
    VER -.->|"re-derives RMSE and RK4 rollout"| EVID
    T5S -.->|"same benchmark, separate script"| DATA
```

**Explanation.** The graph has four LLM-backed nodes and one shared dialogue. The orchestrator's
routing is itself an LLM call. A malformed reply or an illegal `finish` is sent back to the LLM
with the error (at most three attempts). The program never substitutes a route. Only the
executor owns `PythonCodingTool`. Each attempt is written to a new directory and run with a hard
300 s timeout. `finish` is accepted only after an executor turn with `exit_code=0` and a printed
`TEST_RMSE`. While the best result is not below 0.02 V and at least two decisions remain, the
quality gate also rejects `finish`. Task 5 is a separate, hand-written solution to the same
benchmark. It was not produced by the agents. The README keeps the two results apart (Task 5
test RMSE 0.0016 V; Task 6 agent run 1 test RMSE 0.0595 V).

### Figure 2 — HW1 sequence

```mermaid
sequenceDiagram
    actor U as User
    participant M as main()
    participant G as LangGraph runtime
    participant O as Orchestrator
    participant S as Scientist
    participant GA as General
    participant E as Executor
    participant L as OpenAI Responses API
    participant T as PythonCodingTool

    U->>M: python task6/agentic_system.py
    M->>G: graph.stream with dialogue = [user: Task 5 prompt]
    loop at most 8 orchestrator decisions
        G->>O: orchestrator(state)
        O->>L: routing instructions + attempts table + dialogue
        L-->>O: JSON next_role, instruction, reason
        Note over O: invalid JSON or forbidden finish is fed back to the LLM, up to 3 attempts
        alt next_role = scientist
            G->>S: scientist(state)
            S->>L: orchestrator instruction + dialogue
            L-->>S: Duffing model, RK4 and training plan
            S-->>G: append scientist turn
        else next_role = general
            G->>GA: general(state)
            GA->>L: orchestrator instruction + dialogue
            L-->>GA: answer to a side question
            GA-->>G: append general turn
        else next_role = executor
            G->>E: executor(state)
            E->>L: executor contract + latest plan + previous result and source
            L-->>E: complete Python source
            E->>T: write_and_run(new attempt file)
            T-->>E: exit_code, stdout with TRAIN_RMSE and TEST_RMSE, stderr
            E-->>G: append executor turn
        else next_role = finish
            Note over O,G: accepted only after a successful executor run and the quality gate
        end
    end
    G-->>M: final state
    M-->>U: run_summary.json, kept attempt = lowest successful TEST_RMSE
```

**Explanation.** The orchestrator LLM chooses each step from the dialogue. General participation
is optional. In the two recorded validated runs the orchestrator chose only `scientist`,
`executor` and `finish`:
`task6/validated_run`: scientist → executor → finish;
`task6/validated_run_v2`: scientist → executor → scientist → executor → scientist → executor →
executor → finish.
The General branch is shown because the code supports it, not because it was used in those
runs.

---

## 3. HW2 — Human-in-the-loop code approval (Task 1) and simulated optics (Task 2)

### Figure 3 — HW2 architecture

```mermaid
flowchart TB
    subgraph HW1R["Reused without modification: task6/agentic_system.py"]
        BASE["AgenticSilverboxSystem<br/>roles, routing helpers, 8-decision limit"]
        MC2["ModelCaller<br/>OpenAI Responses API"]
        PCT["PythonCodingTool<br/>new file, subprocess, 300 s"]
    end

    subgraph TASK1["HW2 Task 1: HumanInTheLoopSystem in hw2/task1_hitl.py"]
        O1["orchestrator<br/>LLM JSON: next_role, request, instruction, question, reason<br/>direct executor or finish is rejected"]
        S1["scientist, no tools"]
        G1["general, no tools<br/>optional read-only history of Task 5/6 results"]
        P1["executor: prepare<br/>HW1 executor LLM writes the code<br/>_CaptureTool captures it, nothing runs<br/>pending_action with action_id and code SHA-256"]
        X1["executor: execute<br/>only the approved action, once<br/>SHA-256 re-checked, credential-like env vars removed"]
        R1["executor: reject<br/>ACTION_REJECTED, tool not called"]
        FIN1["final_report<br/>built by the program from the best real executor result"]
        CK[("InMemorySaver checkpointer<br/>one thread_id per run")]
        subgraph HB["Human intervention boundary"]
            H1["human node<br/>langgraph.types.interrupt()<br/>clarification, approval or final_approval"]
            OPR["Human operator<br/>reply text + typed approve<br/>resumed with Command(resume=...)"]
        end
    end

    subgraph TASK2["HW2 Task 2: FocusingAgentSystem in hw2/task2_agent.py, SIMULATION ONLY"]
        O2["orchestrator<br/>finish only after a completed experiment<br/>executor refused when run limits are exhausted"]
        S2["scientist, no tools<br/>experiment strategy"]
        G2["general, no tools"]
        E2["executor<br/>LLM proposes all 13 integer settings as JSON<br/>strict check_proposal, up to 3 proposals per turn"]
        FS["FocusingSession.call(arguments)<br/>hw2/focusing_tool.py<br/>one fixed transmission matrix per session"]
        OPT["optimize_focus<br/>hw2/focusing_optimizer.py<br/>binary macropixel flipping + independent validation"]
        BA["BenchAccess<br/>acquire(pattern) only, never reads T"]
        SB["SimulatedBench<br/>hw2/simulated_bench.py<br/>binary amplitude DMD, complex Gaussian T,<br/>shot + read noise camera"]
        FIN2["final report<br/>assembled by the program from validated tool output"]
    end

    BASE --> O1
    BASE --> O2
    O1 -->|"request approval"| P1
    O1 -->|"request clarification or final_approval"| H1
    O1 --> S1
    O1 --> G1
    S1 --> O1
    G1 --> O1
    P1 -->|"awaiting_approval"| H1
    H1 -.->|"interrupt payload: action, code, SHA-256"| OPR
    OPR -.->|"structured decision"| H1
    H1 -->|"approved, same action_id"| X1
    H1 -->|"refused or clarification answered"| O1
    H1 -->|"final measurement approved"| FIN1
    X1 --> PCT
    X1 --> O1
    R1 --> O1
    O1 -.-> CK
    O1 -.-> MC2
    P1 -.-> MC2

    O2 --> S2
    O2 --> G2
    O2 --> E2
    S2 --> O2
    G2 --> O2
    E2 --> FS
    FS --> OPT
    OPT --> BA
    BA --> SB
    SB -.->|"simulated noisy frames"| OPT
    FS -.->|"validated metrics, artifacts"| E2
    E2 --> O2
    O2 -->|"finish"| FIN2
    O2 -.-> MC2
    E2 -.-> MC2
```

**Explanation.** The two HW2 tasks are separate systems on the same HW1 base.

- **Task 1** adds a `human` node that the orchestrator LLM selects itself. Preparation,
  approval and execution are separate graph steps. The executor LLM writes code, but the tool
  call is captured, not executed. The human then judges the exact `action_id` and code SHA-256.
  Only a structured `{"approved": true, "action_id": <same id>}` approves. Anything else refuses
  (fail closed). An approved action runs exactly once and is then marked `executed`. The node
  `executor: reject` is the fail-closed path for any other executor entry.
- **Task 2** keeps the HW1 topology. It replaces the coding tool with one bounded **simulated**
  focusing tool. There is no human approval, no coding tool and no hardware driver. Every
  reported number comes from the tool's independent validation measurements.

### Figure 4 — HW2 Task 1 sequence (recorded refusal, clarification and approval)

This follows the recorded run `task6_runs/hw2_task1_hard_20261012_161518/` (documented in
`submission/HW2_Report_Final.md`, section 4). The user asked for a script that computes the
zero-predictor Test RMSE without training a model.

```mermaid
sequenceDiagram
    actor H as Human operator
    participant D as Run loop (start / resume)
    participant O as Orchestrator
    participant E as Executor
    participant N as human node
    participant L as OpenAI Responses API
    participant T as PythonCodingTool

    D->>O: start(thread_id, user request)
    O->>L: routing decision
    L-->>O: next_role human, request approval
    O->>E: route prepare
    E->>L: HW2 task alignment + HW1 executor contract
    L-->>E: script source
    Note over E: _CaptureTool intercepts write_and_run. action-01 stored as awaiting_approval
    E->>N: route_after_executor
    N-->>H: interrupt(approval: action-01, code, SHA-256)
    Note over D,N: graph paused and checkpointed under thread_id
    H->>N: Command(resume = approved false)
    Note over N: action-01 refused. The coding tool was never called
    N->>O: back to orchestrator
    O->>L: routing decision
    L-->>O: next_role human, request clarification
    O->>N: clarification question
    N-->>H: interrupt(clarification)
    H->>N: resume: still wants the zero-predictor RMSE, asks for a new action to review
    N->>O: human turn appended
    O->>L: routing decision
    L-->>O: next_role human, request approval
    O->>E: route prepare
    E->>L: executor prompt
    L-->>E: new script source
    Note over E: action-02 stored as awaiting_approval
    E->>N: route_after_executor
    N-->>H: interrupt(approval: action-02, code, SHA-256)
    H->>N: resume: approved true, action_id action-02, text Yes, proceed.
    N->>E: action-02 approved
    E->>E: verify SHA-256 and remove credential-like environment variables
    E->>T: write_and_run(approved code), exactly once
    T-->>E: exit_code 0, TEST_RMSE=0.060491854407
    Note over E: status executed. The approval is consumed
    Note over D: the demonstration is bounded and stops after this execution
```

**Explanation.** The diagram shows the real human decisions: a refusal of `action-01`, a
clarification answer, then approval of `action-02` ("Yes, proceed."). According to the recorded
evidence, the refused code never ran (`tool_calls_running_refused_code = 0`). The approved code
ran once, and its hash matched the approved hash. The code also implements the `final_approval`
path (Figure 3), but this recorded demonstration stopped before it. No physical hardware takes
part in HW2.

### Figure 5 — HW2 Task 2 sequence (simulated focusing, recorded live run)

This follows the recorded live run `task6_runs/hw2_task2_live_20261012_211839/`. Orchestrator
decisions: executor → scientist → executor → finish. There were 2 tool calls, both completed.

```mermaid
sequenceDiagram
    actor U as User
    participant O as Orchestrator
    participant S as Scientist
    participant E as Executor
    participant L as OpenAI Responses API
    participant F as FocusingSession
    participant B as SimulatedBench

    U->>O: goal: focus the beam through the simulated scattering medium
    O->>L: routing decision with run limits
    L-->>O: next_role executor
    O->>E: instruction
    E->>L: request, session.describe(), schema, limits, earlier results
    L-->>E: arguments with all 13 settings + rationale
    E->>E: check_proposal: exact keys, integers, minimums, run budget
    E->>F: call(arguments + program-assigned run_name)
    loop iterations + independent validation
        F->>B: acquire(pattern) through BenchAccess
        B-->>F: simulated noisy camera frame
    end
    F-->>E: validated enhancement, focus intensity gain, background suppression
    E-->>O: EXPERIMENT_COMPLETED turn with validated values only
    O->>L: routing decision
    L-->>O: next_role scientist
    O->>S: assess result and propose changes
    S->>L: strategy request
    L-->>S: revised experiment strategy
    S-->>O: scientist turn
    O->>L: routing decision
    L-->>O: next_role executor
    O->>E: run a second experiment
    E->>L: proposal request with first result
    L-->>E: new settings
    E->>F: call(arguments)
    F-->>E: validated results of experiment 2
    E-->>O: EXPERIMENT_COMPLETED turn
    O->>L: routing decision
    L-->>O: next_role finish
    O-->>U: final report assembled by the program from tool output
```

**Explanation.** The executor LLM chooses every experimental setting. The program only
validates the settings (it never fills in or clips them) and assembles the final report from
tool output. All optics here are **simulated**: one fixed random complex Gaussian transmission
matrix, a binary amplitude DMD and a shot/read-noise camera model.

---

## 4. HW3 — LUMEN on the physical iRAYPLE camera

### Figure 6 — HW3 architecture

```mermaid
flowchart TB
    subgraph OPS["Human operator"]
        OP1["Three separate typed inputs:<br/>1. BEGIN FOUR AGENT HARDWARE CHECK<br/>2. APPROVE session_id, within 120 s<br/>3. AUTHORIZE PAID API session_id, within 60 s"]
    end

    subgraph CLI["Trusted application: hw3/physical_lumen_cli.py"]
        C1["main()<br/>--physical-lumen-check, explicit --model,<br/>interactive terminal, first confirmation"]
        C2["preload_vendor_sdk_once()<br/>main thread, before any event loop"]
        C3["physical_mcp_smoke.build_envelope<br/>fresh random session_id"]
        C4["OuterCallBudget<br/>at most 16 outer calls, bounded sizes"]
        C5["_shutdown()<br/>trusted shutdown if not already done,<br/>checks device ownership released"]
    end

    subgraph APPR["Approval plane"]
        ENV["SessionEnvelope, frozen<br/>tools: open, set_exposure, measure, close<br/>1000 to 5000 us, 2 measurements, 7 tool calls, 180 s<br/>canonical JSON + SHA-256"]
        HOST["PhysicalApprovedSessionHost<br/>start, resume, get_mcp_server, shutdown<br/>one device owner per process"]
        HG["hitl_approval graph<br/>START to hardware_approval to END<br/>interrupt(gate.approval_request())<br/>InMemorySaver"]
        GX["Fresh execution-side SessionGate<br/>submit_decision of the same decision"]
    end

    subgraph AGS["LUMEN four-agent graph: hw3/lumen_agent_system.py"]
        LAS["LumenAgentSystem extends HW1 AgenticSilverboxSystem<br/>inherited build(): HW1 topology, 8 decisions<br/>HW1 coding tool disabled"]
        AO["orchestrator<br/>strict JSON, no retry, RoutingError"]
        AS["scientist, no tools<br/>hypothesis and conclusion"]
        AGN["general, no tools<br/>implications and conclusion"]
        AE["executor<br/>first visit: dispatches once<br/>later visit: refused, no device access"]
    end

    subgraph EXP["Adaptive MCP Executor plane"]
        BR["executor_dialogue_bridge<br/>run_adaptive_experiment(): asyncio.run, fail closed<br/>format_executor_turn()"]
        AME["AdaptiveMCPExecutor<br/>decide, validate, one MCP call, record, repeat<br/>parse_decision rejects authority fields"]
        FRG["FirstRunDecisionGuard<br/>exposure only 1000.0 or 5000.0 us<br/>open, then set and measure at most twice, close"]
        BDP["BoundedDecisionProvider<br/>at most 7 calls, compact context,<br/>input and output size limits"]
        OMCH["LumenOpenAIModelCaller, hardware<br/>max 1024 output tokens"]
    end

    OMCO["LumenOpenAIModelCaller, outer agents<br/>max 2048 output tokens"]
    API["OpenAI Responses API, gpt-6-luna<br/>client: max_retries 0, timeout 30 s"]

    subgraph MCPB["Authorized execution boundary"]
        MCL["Official MCP SDK Client<br/>in-process"]
        MCS["create_authorized_server()<br/>MCPServer with 5 camera tools<br/>server-issued action_id"]
        RT["AuthorizedCameraRuntime.execute()<br/>begin_action, one backend call, complete_action<br/>failure: ambiguous, abort, cleanup"]
        GATE["SessionGate, active<br/>allowed tools, exposure range, budgets,<br/>duration, one action in flight"]
        ME["measurements.analyze_frame<br/>mean_dn, saturation_fraction, histogram"]
    end

    subgraph HWL["Physical hardware"]
        IRC["IRaypleCamera, CameraBackend<br/>auto features must read Off, exact exposure readback,<br/>Mono8 only, frames labelled source hardware"]
        SDK["Official HuaRay IMV SDK<br/>IMVApi binding"]
        CAM["iRAYPLE MV-A7200MU130<br/>lensless monochrome sensor, USB3"]
    end

    OP1 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> ENV
    ENV --> HOST
    HOST --> HG
    HG -.->|"pending request: envelope + SHA-256"| OP1
    OP1 -.->|"session_approval decision"| HOST
    HOST --> GX
    GX -->|"graph AND fresh gate accept"| GATE
    HOST -->|"on valid approval builds"| RT
    HOST -->|"on valid approval builds"| MCS
    OP1 -.->|"paid-API consent"| C4
    C4 --> OMCO
    OMCO --> API
    C4 --> LAS
    LAS --> AO
    AO --> AS
    AO --> AGN
    AO --> AE
    AS --> AO
    AGN --> AO
    AE --> BR
    BR --> AME
    AME --> FRG
    FRG --> BDP
    BDP --> OMCH
    OMCH --> API
    AME --> MCL
    MCL --> MCS
    MCS --> RT
    RT --> GATE
    RT --> IRC
    IRC --> SDK
    SDK --> CAM
    CAM -.->|"Mono8 frame"| SDK
    SDK -.->|"pixels copied to NumPy"| IRC
    IRC -.->|"CameraFrame"| RT
    RT --> ME
    ME -.->|"DN metrics"| RT
    RT -.->|"structured tool result"| MCL
    MCL -.->|"history + latest_measurement"| AME
    AME -.->|"result record"| BR
    BR -.->|"LUMEN_EXPERIMENT_COMPLETED turn with measured DN"| AE
    AE -.->|"dialogue turn"| AO
    AME -->|"always at the end: host.shutdown()"| HOST
    C5 -->|"fallback only if not already shut down"| HOST
    IRC -->|"close: restore and verify original ExposureTime,<br/>IMV_Close, IMV_DestroyHandle"| SDK
```

**Explanation.** The LUMEN outer graph is the **original HW1 four-role graph**.
`LumenAgentSystem` inherits `build()` unchanged and replaces only the prompts and the Executor
body. The Orchestrator, Scientist and General have no tools. Nothing in the dialogue grants
authority. Only the Executor can reach the camera, and only once. It does so through
`run_adaptive_experiment()`, which runs `AdaptiveMCPExecutor` on the single approved host.

Inside that loop, the **LLM chooses every MCP action**. Trusted code bounds it in layers:

1. `FirstRunDecisionGuard` restricts the first-run protocol. It refuses but never edits a
   decision.
2. `BoundedDecisionProvider` caps the number of model calls and the context size.
3. `parse_decision` rejects malformed decisions and decisions carrying session, approval or hash
   fields.
4. The **`SessionGate`**, which a human approved through LangGraph and which is re-validated by a
   fresh execution-side gate, authorizes or refuses each call. A refused call never reaches the
   backend.
5. `IRaypleCamera` adds device-level checks. Auto features must read `Off`, the exposure readback
   must match the request exactly (nothing is clamped), and frames must be Mono8.

Each measurement's DN statistics go back into the executor's history and `latest_measurement`
before the next model decision. That is the physical feedback loop. Shutdown always runs:
`AdaptiveMCPExecutor` calls `host.shutdown()` in a `finally` block, and the CLI repeats it only
if the host is not already shut down. On close, `IRaypleCamera` restores the original
`ExposureTime` and verifies it by readback.

### Figure 7 — HW3 sequence (four-agent physical run)

```mermaid
sequenceDiagram
    actor Op as Operator
    participant CLI as physical_lumen_cli
    participant Host as PhysicalApprovedSessionHost
    participant HG as Approval graph
    participant Gate as SessionGate
    participant Orc as Orchestrator
    participant Sci as Scientist
    participant Gen as General
    participant Exe as Executor and bridge
    participant AME as AdaptiveMCPExecutor
    participant Guard as Guard and BoundedDecisionProvider
    participant API as OpenAI API
    participant MCP as MCP server and runtime
    participant Cam as IRaypleCamera, SDK, sensor

    Op->>CLI: --physical-lumen-check --model gpt-6-luna
    CLI-->>Op: hardware and paid-API warning
    Op->>CLI: BEGIN FOUR AGENT HARDWARE CHECK
    Note over CLI: this confirmation approves neither hardware nor API use
    CLI->>CLI: preload_vendor_sdk_once() on the main thread
    CLI->>CLI: build_envelope(new_session_id())
    CLI->>Host: construct with envelope and SDK binding, no SDK call
    CLI->>Host: start(session_id)
    Host->>HG: start_approval(thread_id = session_id)
    HG-->>Host: interrupt(approval_request: envelope + SHA-256)
    Host-->>CLI: pending request
    CLI->>CLI: pending request must equal the trusted envelope
    CLI-->>Op: show envelope, SHA-256, limits, first-run restriction
    Op->>CLI: APPROVE session_id
    CLI->>Host: resume(session_approval(session_id, sha256, true))
    Host->>HG: Command(resume = decision)
    HG-->>Host: outcome approved
    Host->>Gate: fresh SessionGate.submit_decision(same decision)
    Gate-->>Host: active, session clock starts
    Host->>Host: build AuthorizedCameraRuntime and authorized MCP server
    Host-->>CLI: activated true
    CLI-->>Op: separate paid-API consent request
    Op->>CLI: AUTHORIZE PAID API session_id
    CLI->>CLI: create OpenAI client, OuterCallBudget, BoundedDecisionProvider, Guard
    CLI->>Orc: graph.stream with user task LUMEN_TASK

    Note over Orc,API: route below is the verified run, chosen step by step by the orchestrator LLM
    Orc->>API: routing decision 1
    API-->>Orc: next_role scientist
    Orc->>Sci: instruction
    Sci->>API: hypothesis request
    API-->>Sci: hypothesis and research goal
    Sci-->>Orc: scientist turn

    Orc->>API: routing decision 2
    API-->>Orc: next_role general
    Orc->>Gen: instruction
    Gen->>API: analysis request
    API-->>Gen: implications and constraints
    Gen-->>Orc: general turn

    Orc->>API: routing decision 3
    API-->>Orc: next_role executor
    Orc->>Exe: executor node, first visit
    Exe->>AME: run_adaptive_experiment(host, guard, goal, 7) in asyncio.run
    AME->>MCP: Client on get_mcp_server(), list_tools

    AME->>Guard: decision 1 context, no measurement yet
    Guard->>API: bounded request
    API-->>Guard: call_tool camera_open
    Guard-->>AME: forwarded unchanged, check passed
    AME->>MCP: call_tool camera_open
    MCP->>Gate: begin_action(camera_open)
    MCP->>Cam: open, CreateHandle, check auto features Off, read ExposureTime
    Cam-->>MCP: opened
    MCP->>Gate: complete_action succeeded

    AME->>Guard: decision 2 context
    Guard->>API: bounded request
    API-->>Guard: camera_set_exposure 1000.0
    Guard-->>AME: allowed candidate, check passed
    AME->>MCP: call_tool camera_set_exposure 1000.0
    MCP->>Gate: begin_action, 1000.0 within approved range
    MCP->>Cam: IMV_SetDoubleFeatureValue ExposureTime, exact readback
    Cam-->>MCP: applied 1000.0 us

    AME->>Guard: decision 3 context
    Guard->>API: bounded request
    API-->>Guard: call_tool camera_measure
    AME->>MCP: call_tool camera_measure
    MCP->>Gate: begin_action, measurement 1 of 2
    MCP->>Cam: StartGrabbing, IMV_GetFrame, validate Mono8, copy, release, stop
    Cam-->>MCP: CameraFrame frame_id 1, source hardware
    MCP->>MCP: analyze_frame
    MCP-->>AME: measurement 1, mean 15.5104 DN
    Note over AME: measurement 1 stored in history and latest_measurement

    AME->>Guard: decision 4 context includes measurement 1
    Guard->>API: bounded request with 15.5104 DN
    API-->>Guard: camera_set_exposure 5000.0
    Note over Guard,API: second exposure chosen by the model after receiving measurement 1
    AME->>MCP: call_tool camera_set_exposure 5000.0
    MCP->>Gate: begin_action, within approved range
    MCP->>Cam: write ExposureTime, exact readback
    Cam-->>MCP: applied 5000.0 us

    AME->>Guard: decision 5 context
    Guard->>API: bounded request
    API-->>Guard: call_tool camera_measure
    AME->>MCP: call_tool camera_measure
    MCP->>Gate: begin_action, measurement 2 of 2
    MCP->>Cam: acquire one Mono8 frame
    Cam-->>MCP: CameraFrame frame_id 2, source hardware
    MCP-->>AME: measurement 2, mean 79.4017 DN

    AME->>Guard: decision 6 context
    Guard->>API: bounded request
    API-->>Guard: call_tool camera_close
    AME->>MCP: call_tool camera_close
    MCP->>Gate: begin_action(camera_close)
    MCP->>Cam: close: restore original ExposureTime, verify readback, IMV_Close, IMV_DestroyHandle
    Cam-->>MCP: closed, restoration verified

    AME->>Guard: decision 7 context
    Guard->>API: bounded request
    API-->>Guard: finish with reason
    AME->>Host: shutdown() in finally
    Host->>Gate: runtime.shutdown: gate.close()
    Host->>Cam: backend.close(), native Close and DestroyHandle not repeated
    AME-->>Exe: result: status completed, sources hardware
    Exe->>Exe: format_executor_turn gives LUMEN_EXPERIMENT_COMPLETED
    Exe-->>Orc: executor turn with frame ids, exposures and DN metrics

    Orc->>API: routing decision 4
    API-->>Orc: next_role general
    Orc->>Gen: interpret the measured DN values
    Gen->>API: conclusion request
    API-->>Gen: conclusion based only on measured DN
    Gen-->>Orc: post-experiment conclusion turn
    Orc->>API: routing decision 5
    API-->>Orc: next_role finish
    Orc-->>CLI: graph reaches END
    CLI->>Host: status(): already shut down, device ownership released
    CLI-->>Op: report, PHYSICAL_LUMEN_CHECK PASS, exit code 0
```

**Explanation.** The diagram has three human gates in a fixed order: a typed confirmation, a
real LangGraph approval of the exact envelope, and a separate paid-API consent. The OpenAI
client is created only after all three succeed.

- **Hardware decisions.** The six MCP tool calls and the final `finish` are seven decisions
  made by the hardware model. This is exactly the D5A budget (`MAX_MODEL_DECISIONS = 7`), with
  no slack. The first-run guard allows only the order open → (set → measure) × 2 → close. A
  measurement is allowed only after a newly set exposure. So the second exposure decision can
  only be made after measurement 1 is in the executor's history. The `PASS` verdict checks this
  explicitly (`feedback_evidence`).
- **Outer order.** The diagram shows the verified physical run. Its role visit order was
  orchestrator → scientist → orchestrator → general → orchestrator → executor → orchestrator →
  general → orchestrator. Each step was chosen by an orchestrator LLM routing decision; the route
  is not hardcoded and the code permits other orders. The `PASS` verdict guarantees that all four
  roles took part, the Executor dispatched exactly once, and a Scientist or General conclusion
  followed the completed Executor turn. In this run the post-experiment conclusion was written
  by **General**.
- **API requests.** 5 Orchestrator + 1 Scientist + 2 General = 8 outer-agent requests, plus 7
  hardware Executor decisions, for 15 OpenAI API requests in total.
- **Restoration.** Restoration happens in the model-selected `camera_close`. The trusted
  `host.shutdown()` then closes the gate and releases the backend without repeating the native
  close calls.

---

## 5. Source traceability

| Diagram component | Figure(s) | Source file(s) |
|---|---|---|
| `AgenticSilverboxSystem`, `DialogueState`, `build()`, routing, quality gate, `MAX_DECISIONS = 8` | 1, 2, 3, 6 | `task6/agentic_system.py` |
| `ModelCaller` (OpenAI Responses API) | 1, 2, 3 | `task6/agentic_system.py` |
| `PythonCodingTool.write_and_run` (new attempt directory, 300 s timeout) | 1, 2, 3, 4 | `task6/agentic_system.py` |
| Task 5 prompt / hand-written solution / recorded result | 1 | `task6/task5_assignment.txt`, `task5/task5_silverbox_pinn.py`, `task5/task5_results.json` |
| Independent HW1 verifier (checks A–D) | 1 | `task6/verify_run.py` |
| Recorded HW1 runs (decision sequences, RMSE) | 2 | `task6/validated_run/run_summary.json`, `task6/validated_run_v2/run_summary.json` |
| `HumanInTheLoopSystem`, `human` node, `interrupt()`, `Command(resume=...)`, `pending_action`, `_CaptureTool`, `secrets_removed_from_environment`, `final_report` | 3, 4 | `hw2/task1_hitl.py` |
| Recorded refusal / clarification / approval demonstration | 4 | `hw2/demo_task1_hard.py`; evidence summarised in `submission/HW2_Report_Final.md` §4 |
| `FocusingAgentSystem`, `check_proposal`, run limits, program-built final report | 3, 5 | `hw2/task2_agent.py`, `hw2/demo_task2_live.py` |
| `FocusingSession.call` | 3, 5 | `hw2/focusing_tool.py` |
| `optimize_focus`, `BenchAccess` | 3, 5 | `hw2/focusing_optimizer.py` |
| `SimulatedBench` (DMD, transmission matrix *T*, noise model) | 3, 5 | `hw2/simulated_bench.py` |
| Trusted CLI, three human gates, `OuterCallBudget`, verdict, `_shutdown` | 6, 7 | `hw3/physical_lumen_cli.py` |
| `FirstRunDecisionGuard`, `EXPOSURE_CANDIDATES_US = (1000.0, 5000.0)`, `MAX_MODEL_DECISIONS = 7`, approval/consent prompts, `feedback_evidence` | 6, 7 | `hw3/physical_adaptive_cli.py` |
| Session envelope values (4 tools, 1000–5000 µs, 2 measurements, 7 tool calls, 180 s) | 6, 7 | `hw3/physical_mcp_smoke.py` (`build_envelope`, `new_session_id`) |
| `LumenAgentSystem`, strict routing, single dispatch | 6, 7 | `hw3/lumen_agent_system.py` |
| `PhysicalApprovedSessionHost`, `preload_vendor_sdk_once`, `PHYSICAL_LIMITS`, device ownership | 6, 7 | `hw3/physical_approved_session_host.py` |
| `start` / `resume` / `get_mcp_server` / `shutdown`, dual validation (graph + fresh gate) | 6, 7 | `hw3/approved_session_host.py` |
| Approval graph `START → hardware_approval → END` | 6, 7 | `hw3/hitl_approval.py` |
| `SessionEnvelope`, SHA-256, `SessionGate` state machine and budgets | 6, 7 | `hw3/session_policy.py` |
| `AuthorizedCameraRuntime.execute`, ambiguous-outcome abort, `shutdown` | 6, 7 | `hw3/authorized_camera.py` |
| Authorized MCP server, server-issued `action_id` | 6, 7 | `hw3/authorized_mcp.py` |
| `IRaypleCamera` (open, exposure write and readback, Mono8 capture, restore on close) | 6, 7 | `hw3/irayple_camera.py` |
| `AdaptiveMCPExecutor`, `parse_decision`, event history, `latest_measurement` | 6, 7 | `hw3/executor_mcp_loop.py` |
| `run_adaptive_experiment`, `format_executor_turn`, `LUMEN_EXPERIMENT_COMPLETED` | 6, 7 | `hw3/executor_dialogue_bridge.py` |
| `BoundedDecisionProvider` | 6, 7 | `hw3/llm_decision_provider.py` |
| `LumenOpenAIModelCaller`, `create_live_openai_client` (max_retries 0, 30 s) | 6, 7 | `hw3/openai_model_caller.py` |
| `analyze_frame` (DN metrics) | 6, 7 | `hw3/measurements.py` |
| `CameraBackend`, `CameraFrame`, source labels | 6 | `hw3/camera_contract.py` |

---

## 6. Physical experimental evidence (HW3)

The operator reported the following outcome of the four-agent physical run
(`hw3/physical_lumen_cli.py`) on the real iRAYPLE MV-A7200MU130:

| Item | Observed value |
|---|---|
| Verdict / exit code | `PASS`, exit code 0 |
| Agent participation | Orchestrator, Scientist, General and Executor all took part. Role visit order: orchestrator → scientist → orchestrator → general → orchestrator → executor → orchestrator → general → orchestrator |
| Model | `gpt-6-luna` |
| Real API calls | Reported by the terminal: 8 outer-agent requests (5 Orchestrator, 1 Scientist, 2 General) + 7 hardware Executor requests = **15** in total |
| Authorized MCP tool calls | 6 (open, set exposure, measure, set exposure, measure, close) |
| Physical measurements | 2 Mono8 frames, source `hardware` |
| Measurement 1 | exposure 1000 µs, mean **15.5104 DN** |
| Measurement 2 | exposure 5000 µs, mean **79.4017 DN** |
| Adaptivity | the second exposure decision was made after measurement 1 had been received |
| Restoration and shutdown | succeeded. MV Viewer independently confirmed that `ExposureTime` returned to 5000 µs |
| Recording | the operator recorded the live demonstration: `HW3_03_Four_Agent_LUMEN_D6A.mp4` (inspected) |

The measured values came from the physical sensor, not from `FakeCamera`:

- **Source labels.** `PhysicalApprovedSessionHost` constructs only an `IRaypleCamera` backend.
  `AuthorizedCameraRuntime` rejects any frame whose source differs from the backend's declared
  source.
- **Verdict check.** The PASS verdict requires the observed sources to be exactly `["hardware"]`.

**Three successful physical experiments.** The four-agent run above (D6A) is one of three
successful physical experiments. The other two are:

- **D4:** a fixed six-call MCP diagnostic, `hw3/physical_mcp_smoke.py`, which is not LLM-driven.
- **D5A:** an LLM-driven adaptive MCP experiment, `hw3/physical_adaptive_cli.py`.

D5A and D6A each contain exactly one feedback-driven exposure correction. All values and their
evidence sources are given in `hw3/docs/physical_experiments.md`.

Screenshots and video recordings exist as separate files in the repository root:

| Recording | Evidence status |
|---|---|
| `HW3_01_Physical_MCP_D4.mp4` | Successful D4 run according to the operator; not independently inspected |
| `HW3_02_Adaptive_LLM_D5A.mp4` | Inspected |
| `HW3_03_Four_Agent_LUMEN_D6A.mp4` | Inspected |

Screenshots include `mvviewer screenshot.png`. `demo2 hw3.mp4` is the HW2 *simulated* optics demo
and is not HW3 hardware evidence. **Media files are not embedded in this document.** No image, raw
frame or report file was saved by the runs themselves (by design of the CLIs).

**Rendered figures.** All eight figures are available as SVG files in `hw3/docs/figures/`:

| Figure | File |
|---|---|
| 1 | `hw1_architecture.svg` |
| 2 | `hw1_sequence.svg` |
| 3 | `hw2_architecture.svg` |
| 4 | `hw2_hitl_sequence.svg` |
| 5 | `hw2_simulation_sequence.svg` |
| 6 | `hw3_architecture.svg` |
| 7 | `hw3_sequence.svg` |
| 8 | `hw1_hw2_hw3_overview.svg` (integrated HW1/HW2/HW3 architecture, Section 9) |

## 7. What was simulated, offline-tested and verified on hardware

| Component | Simulated | Offline-tested (mocks/fakes) | Verified on physical hardware |
|---|---|---|---|
| HW1 Silverbox modelling (PINN / neural state-space, RK4) | No (recorded benchmark data, not simulation of hardware) | Independent verifier `task6/verify_run.py` | Not applicable, no hardware |
| HW2 Task 1 HITL code approval | No optics involved | `hw2/tests/test_task1_hitl.py` | Not applicable. Real LLM and real human decisions, no hardware |
| HW2 Task 2 focusing experiment | **Yes**: DMD, transmission matrix, camera noise | `hw2/tests/` (bench, optimizer, tool, agent) | **No** |
| HW3 approval graph, SessionGate, runtime, MCP server, executor loop, bridge | Offline path uses `FakeCamera` (labelled synthetic) | `hw3/tests/` (baseline stated as 1004 passing tests) | **Yes**, in the four-agent run above |
| HW3 `IRaypleCamera` + HuaRay IMV SDK | No | `hw3/tests/` with a mocked SDK binding | **Yes**: open, exposure write/readback, two Mono8 frames, restoration |
| HW3 LLM-driven adaptive exposure choice | No | Scripted providers in `hw3/tests/` | **Yes**: real `gpt-6-luna` decisions on real measurements |

## 8. Scientific scope and limitations

- **Distinct systems.** HW1's PINN/Silverbox system works on a recorded electrical benchmark
  (`SNLS80mV.mat`). It is unrelated to the HW3 camera.
- **HW2 optics are simulated.** No laser, DMD, scattering sample or camera was used. HW2
  enhancement values are properties of the simulator.
- **HW3 sensor.** HW3 uses a real, **lensless** monochrome sensor. No lens was mounted, so **no
  optical focus or image formation** has been demonstrated.
- **Uncalibrated DN.** HW3 values are **uncalibrated digital numbers** from Mono8 frames. No
  absolute irradiance, no calibrated sensitivity, no dark-frame correction and **no linearity**
  of DN versus exposure has been demonstrated. Two points at different exposures are not a
  linearity test. The illumination was ambient and was not measured.
- **Restricted exposure set.** The first physical adaptive experiment restricts exposure
  candidates to **1000 µs and 5000 µs** (`FirstRunDecisionGuard`). This is narrower than the
  approved 1000–5000 µs envelope, which is in turn narrower than the SDK range. The model could
  choose only between these two values. The result is therefore a demonstration of
  feedback-driven selection, not of fine exposure optimisation. The second measurement
  (79.4 DN) is near, but not exactly at, the 70 DN target.
- **Limits are enforced by trusted code.** The LLM chooses actions, but trusted code enforces
  the physical limits: the guard, `parse_decision`, the human-approved `SessionGate`, and the
  device checks in `IRaypleCamera`. The model cannot widen the envelope, supply an `action_id`,
  or approve a session.
- **Crash limitation.** A forcibly killed process cannot restore `ExposureTime`. The CLI states
  this and advises checking it in MV Viewer.
- **Run-order source.** The order of the two measurements (1000 µs first, then 5000 µs) and the
  numerical values in Section 6 come from the operator's run report. No run log is stored in the
  repository.
- **Rendering status.** All seven Mermaid figures were rendered with Mermaid CLI 11.16.0 in
  iteration HW3.8-C2, without any change to their source. The resulting SVG files are in
  `hw3/docs/figures/`. All seven were validated as well-formed XML with a valid viewBox, and they
  were inspected visually.

---

## 9. Integrated whole-system architecture (HW1 → HW2 → HW3)

### Figure 8 — Integrated HW1/HW2/HW3 System Architecture

Rendered SVG: [`figures/hw1_hw2_hw3_overview.svg`](figures/hw1_hw2_hw3_overview.svg)

```mermaid
flowchart TB
    subgraph HW1["HW1 (Tasks 5/6): shared origin, task6/agentic_system.py"]
        H1G["LangGraph StateGraph with one shared dialogue<br/>MAX_DECISIONS = 8, every role is an LLM<br/>(OpenAI Responses API)"]
        H1O["Orchestrator<br/>LLM routing decision"]
        H1S["Scientist<br/>no tools"]
        H1N["General<br/>no tools"]
        H1E["Executor<br/>the only role with a tool"]
        H1T["PythonCodingTool<br/>new script, subprocess, 300 s timeout"]
        H1D[("Silverbox benchmark SNLS80mV.mat<br/>recorded data, numerical modelling only")]
        H1C["class AgenticSilverboxSystem<br/>base class reused, never modified"]
        H1G --> H1O
        H1O <--> H1S
        H1O <--> H1N
        H1O <--> H1E
        H1E --> H1T
        H1T -->|"generated script reads"| H1D
        H1D ~~~ H1C
    end

    NOTE["Reading guide: one shared HW1 origin, three SEPARATE applications.<br/>Each is started on its own; they never run in one runtime and<br/>share no session, approval or hardware. The HW2 bench is<br/>SIMULATION ONLY and is not connected to the HW3 camera."]

    subgraph APP1["HW2 Task 1: HumanInTheLoopSystem"]
        A1R["Separate application: hw2/task1_hitl.py<br/>four HW1 roles + added human node<br/>subclass of AgenticSilverboxSystem"]
        A1P["Executor: prepare<br/>LLM writes code; captured, not run<br/>pending action_id + code SHA-256"]
        A1H{{"HUMAN STOP: code approval<br/>human node, LangGraph interrupt()<br/>resumed with Command(resume=...)"}}
        A1OP["Human operator<br/>approve, refuse or clarify"]
        A1X["Executor: execute<br/>approved action only, exactly once,<br/>SHA-256 re-checked"]
        A1T["PythonCodingTool<br/>reused from HW1, Silverbox data"]
        A1R --> A1P
        A1P -->|"awaiting approval"| A1H
        A1H <-.-> A1OP
        A1H -->|"approved, same action_id"| A1X
        A1H -->|"refused: tool never called"| A1R
        A1X --> A1T
    end

    subgraph APP2["HW2 Task 2: SIMULATION ONLY"]
        A2R["Separate application: hw2/task2_agent.py<br/>FocusingAgentSystem, four HW1 roles<br/>subclass of AgenticSilverboxSystem<br/>coding tool removed"]
        A2E["Executor<br/>LLM proposes all 13 integer settings<br/>check_proposal validates, never edits"]
        A2F["Simulated focusing tool<br/>FocusingSession.call + optimize_focus"]
        A2B["SimulatedBench: SIMULATION ONLY<br/>simulated binary DMD,<br/>random complex transmission matrix T,<br/>simulated camera with shot and read noise"]
        A2R --> A2E
        A2E --> A2F
        A2F -->|"acquire(pattern) via BenchAccess"| A2B
        A2B -.->|"simulated frames"| A2F
        A2F -.->|"validated metrics"| A2E
    end

    subgraph APP3["HW3 LUMEN, separate application: hw3/physical_lumen_cli.py. PHYSICAL HARDWARE"]
        subgraph A3AG["LumenAgentSystem: subclass of AgenticSilverboxSystem, inherited build(), 8 decisions, coding tool disabled"]
            A3O["Orchestrator<br/>no physical tools"]
            A3S["Scientist<br/>no physical tools"]
            A3G["General<br/>no physical tools"]
            A3E["Executor<br/>the only hardware-operating role"]
            A3O <--> A3S
            A3O <--> A3G
            A3O <--> A3E
        end
        A3BR["executor_dialogue_bridge<br/>run_adaptive_experiment(), one dispatch"]
        A3AME["AdaptiveMCPExecutor<br/>the LLM selects each physical MCP command<br/>FirstRunDecisionGuard + BoundedDecisionProvider"]
        subgraph A3MCP["MCP connection and authorized MCP server"]
            A3MCL["Official MCP SDK Client"]
            A3MCS["Authorized MCP server<br/>create_authorized_server(), 5 camera tools,<br/>server-issued action_id"]
            A3RT["AuthorizedCameraRuntime<br/>one backend call per authorized action"]
            A3ME["analyze_frame<br/>raw, uncalibrated Mono8 DN"]
        end
        subgraph A3AP["Human hardware-session approval: separate implementation from HW2 code approval"]
            A3OP["Human operator<br/>typed APPROVE session_id"]
            A3ENV["SessionEnvelope, canonical JSON + SHA-256<br/>4 tools, 1000 to 5000 us,<br/>2 measurements, 7 tool calls, 180 s"]
            A3HG{{"HUMAN STOP: hardware approval<br/>hitl_approval graph, hardware_approval node<br/>LangGraph interrupt() / Command(resume=...)"}}
            A3GATE["SessionGate<br/>authorizes or refuses every MCP call"]
        end
        subgraph A3HW["Camera connector and physical hardware"]
            A3IRC["IRaypleCamera connector (CameraBackend)<br/>auto features Off, exact exposure readback,<br/>Mono8 only, frames labelled source hardware"]
            A3SDK["Official HuaRay IMV SDK<br/>IMVApi binding, MVSDKmd.dll"]
            A3CAM["REAL iRAYPLE MV-A7200MU130<br/>lensless monochrome sensor, USB3"]
        end
        A3CL["Trusted cleanup, always runs<br/>host.shutdown(): gate closed, device released;<br/>on close the original ExposureTime<br/>is restored and verified"]

        A3E ==>|"HARDWARE TURN"| A3BR
        A3BR --> A3AME
        A3AME -->|"call_tool"| A3MCL
        A3MCL --> A3MCS
        A3MCS --> A3RT
        A3ENV --> A3HG
        A3OP <-.-> A3HG
        A3HG -->|"approved: activates"| A3GATE
        A3RT -->|"begin_action / complete_action"| A3GATE
        A3RT --> A3IRC
        A3IRC --> A3SDK
        A3SDK --> A3CAM
        A3CAM -.->|"Mono8 frame"| A3SDK
        A3SDK -.->|"pixels copied to NumPy"| A3IRC
        A3IRC -.->|"CameraFrame"| A3RT
        A3RT --> A3ME
        A3ME -.->|"tool result with measured DN"| A3MCL
        A3MCL -.->|"FEEDBACK: latest_measurement<br/>informs the next LLM decision"| A3AME
        A3AME -.->|"LUMEN_EXPERIMENT_COMPLETED<br/>turn with measured DN"| A3E
        A3AME -->|"finally"| A3CL
        A3CL -->|"close and restore"| A3IRC
    end

    H1C -.->|"extends: physical MCP path"| A3O
    H1C -.->|"extends: adds human stop"| A1R
    H1C -.->|"extends: simulated tool"| A2R
    A1T ~~~ A3O
    A2B ~~~ A3O

    classDef sim fill:#fff4cc,stroke:#b8860b,stroke-width:2px,stroke-dasharray:6 4,color:#000
    classDef hw fill:#dff3e4,stroke:#1e7b34,stroke-width:2px,color:#000
    classDef human fill:#ffe0cc,stroke:#c0501a,stroke-width:2px,color:#000
    classDef note fill:#f2f2f2,stroke:#555,stroke-width:1px,color:#000
    classDef base fill:#e6e6ff,stroke:#333,stroke-width:2px,color:#000
    class A2B,A2F sim
    class A3IRC,A3SDK,A3CAM hw
    class A1H,A1OP,A3HG,A3OP human
    class NOTE note
    class H1C base
```

**Figure 8 caption.** Integrated architecture of the complete system after Homework 3. The HW1
four-role LangGraph system (`AgenticSilverboxSystem`: Orchestrator, Scientist, General and
Executor, with `PythonCodingTool`) is the common origin. Three *separate* applications subclass
it: the HW2 Task 1 human-in-the-loop code-approval system, the HW2 Task 2 focusing agent on a
**simulated** optical bench, and the HW3 LUMEN system, which drives a **real** iRAYPLE camera
through an authorized MCP server and the `IRaypleCamera` connector. Dashed arrows from the base
class denote inheritance, not runtime data flow. No edge connects the three applications to each
other. Human-stop nodes are orange, simulated components are yellow with dashed borders, and
physical hardware components are green.

**The three architectural stages.**

1. **HW1: the original system.** A LangGraph `StateGraph` over one shared dialogue. An
   Orchestrator LLM routes between the Scientist, the General and the Executor, for at most
   eight decisions. Only the Executor has a tool, `PythonCodingTool`, which writes and runs a new
   Python script for the Silverbox modelling task on recorded benchmark data (`SNLS80mV.mat`).
   No hardware is involved.
2. **HW2: two independent extensions.**
   - *Task 1, the human stop.* `HumanInTheLoopSystem` adds a `human` node that pauses the graph
     with LangGraph `interrupt()` and resumes with `Command(resume=...)`. The Executor's code is
     first captured, not run, and bound to an `action_id` and a SHA-256 hash. Only an approved
     action with the same `action_id` and hash runs, exactly once, through the reused HW1
     `PythonCodingTool`.
   - *Task 2, the simulated bench.* `FocusingAgentSystem` removes the coding tool. Its Executor
     proposes experiment settings for `FocusingSession`, which drives `SimulatedBench` (a
     simulated binary DMD, a random complex transmission matrix and a simulated noisy camera).
     **This bench is SIMULATION ONLY.** It has no hardware driver and is not connected to the
     HW3 camera.
3. **HW3: the physical LUMEN extension.** `LumenAgentSystem` inherits the HW1 `build()` and
   disables the coding tool. The Orchestrator, Scientist and General have no physical tools. The
   Executor is the only hardware-operating role. Its single **hardware turn** dispatches through
   `executor_dialogue_bridge` to `AdaptiveMCPExecutor`, where an LLM selects each physical MCP
   command, bounded by `FirstRunDecisionGuard` and `BoundedDecisionProvider`. Every call passes
   through the official MCP SDK client to the authorized MCP server and `AuthorizedCameraRuntime`.
   A `SessionGate`, activated only after a human approves the exact `SessionEnvelope` in the
   LangGraph `hardware_approval` interrupt, authorizes or refuses each call. The `IRaypleCamera`
   connector drives the real iRAYPLE MV-A7200MU130 lensless monochrome sensor through the
   official HuaRay IMV SDK. Each measurement returns raw, uncalibrated Mono8 DN, which feeds the
   next model decision. Trusted cleanup (`host.shutdown()`) always runs, and on close the
   original `ExposureTime` is restored and verified by readback.

**Simulation versus hardware.** HW2 Task 2 is **simulated**: its DMD, transmission matrix and
camera exist only in software (`hw2/simulated_bench.py`). HW3 is **physical**: its measurements
come from a real sensor and are labelled `source=hardware`. The two human stops are distinct
implementations. HW2 approves a *code action* by `action_id` and code SHA-256 (`hw2/task1_hitl.py`).
HW3 approves a *hardware session envelope* by session ID and envelope SHA-256 and then enforces it
with `SessionGate` (`hw3/hitl_approval.py`, `hw3/session_policy.py`). Both use LangGraph
`interrupt()` / `Command(resume=...)`, but they share no code path or approval state. The
physical results and limitations in Sections 6–8 apply unchanged. Figure 8 adds no new
experimental claim.

**Source traceability for Figure 8.**

| Figure 8 component | Source file(s) |
|---|---|
| `AgenticSilverboxSystem`, four roles, `StateGraph`, `MAX_DECISIONS = 8`, `PythonCodingTool` | `task6/agentic_system.py` |
| `HumanInTheLoopSystem`, `human` node, `interrupt()` / `Command(resume=...)`, captured action, code SHA-256, single approved execution | `hw2/task1_hitl.py` |
| `FocusingAgentSystem` (coding tool removed), Executor proposal check | `hw2/task2_agent.py` |
| `FocusingSession.call`, `optimize_focus`, `BenchAccess` | `hw2/focusing_tool.py`, `hw2/focusing_optimizer.py` |
| `SimulatedBench`, simulated DMD, transmission matrix, noise model | `hw2/simulated_bench.py` |
| `LumenAgentSystem` (inherited `build()`, coding tool disabled, single Executor dispatch) | `hw3/lumen_agent_system.py` |
| Separate HW3 application entry point | `hw3/physical_lumen_cli.py` |
| `run_adaptive_experiment`, `LUMEN_EXPERIMENT_COMPLETED` turn | `hw3/executor_dialogue_bridge.py` |
| `AdaptiveMCPExecutor`, MCP SDK `Client`, `latest_measurement` feedback | `hw3/executor_mcp_loop.py` |
| `FirstRunDecisionGuard` / `BoundedDecisionProvider` | `hw3/physical_adaptive_cli.py`, `hw3/llm_decision_provider.py` |
| `hardware_approval` interrupt graph | `hw3/hitl_approval.py` |
| `SessionEnvelope`, envelope SHA-256, `SessionGate` | `hw3/session_policy.py` |
| Approved host, `host.shutdown()` | `hw3/physical_approved_session_host.py`, `hw3/approved_session_host.py` |
| Authorized MCP server (`create_authorized_server`, server-issued `action_id`) | `hw3/authorized_mcp.py` |
| `AuthorizedCameraRuntime` (`begin_action` / `complete_action`) | `hw3/authorized_camera.py` |
| `analyze_frame` (raw DN metrics) | `hw3/measurements.py` |
| `IRaypleCamera` connector, exposure readback, Mono8, restoration on close | `hw3/irayple_camera.py` |
| HuaRay IMV SDK path, `MVSDKmd.dll`, expected model MV-A7200MU130 | `hw3/irayple_discovery.py` |

**Rendering status.** Figure 8 was rendered from the Mermaid source above with the installed
Mermaid CLI 11.16.0 (`@mermaid-js/mermaid-cli`, bundled `mermaid` 11.17.2) in iteration
HW3.11-A, using local Google Chrome through Puppeteer and a white background. The output
`hw3/docs/figures/hw1_hw2_hw3_overview.svg` is well-formed XML and was inspected visually in
Chrome. Figures 1–7 and their SVG files were not changed.
