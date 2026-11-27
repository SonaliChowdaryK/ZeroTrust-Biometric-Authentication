# Integration and Smoke-Test Report

**Scope:** Package-level functional checks only. No real-user biometric acceptance testing was performed.

## Policy and application
- Frozen policy unit tests: **9/9 passed**.
- Python compile check: **passed**.
- Gradio app import: **passed**.
- Local Gradio HTTP launch: **HTTP 200**.

## Model checks
| Module | Check | Result |
|---|---|---|
| M1 Face Anti-Spoofing | Checkpoint load + inference on supplied dataset sample labeled ClientFace | PASS; Live/ACCEPT, confidence 0.5889 |
| M1 Face Anti-Spoofing | Inference on supplied dataset sample labeled ImposterFace | PASS; Spoof/REJECT, confidence 0.9710 |
| M2 Face Recognition | ArcFace ONNX file and enrollment template inspected | Assets PASS; ONNX 87,206,462 bytes; template shape (512,) |
| M2 Face Recognition | Real inference in this package-test runtime | NOT RUN: `onnxruntime`/`insightface` are not installed in this test environment. The adapter is prepared; runtime dependency install and actual face verification remain to be tested. |
| M3 Speaker Recognition | Local ECAPA checkpoints and enrollment template loaded; synthetic 4-second tone inference | PASS as functional smoke; synthetic tone rejected (similarity 0.0645) |
| M4 Voice Anti-Spoofing | Checkpoint loaded; synthetic 4-second tone inference | PASS as functional smoke; synthetic tone classified SPOOF (spoof probability 0.6543) |

## Full pipeline smoke
Using one supplied dataset face sample and a generated synthetic tone:
- M1: ACCEPT
- M2: ERROR / UNKNOWN (missing ONNX Runtime in this test environment)
- M3: REJECT (synthetic audio)
- M4: REJECT (synthetic audio)
- Central decision: **DENY**, because a valid completed rejection takes precedence under the frozen policy.

This is an expected fail-closed result. It is **not** a successful genuine-user authentication.

## Known issues and risks
1. M1 validation report indicates BPCER around 59.78%, meaning genuine samples were often rejected in that reported evaluation. This is a significant usability/accuracy issue.
2. M2 inference must be run after installing `onnxruntime`, `insightface`, and initializing the detector assets; a real face sample is needed.
3. M3/M4 need genuine and impostor/replay/synthetic real-world test samples for meaningful evaluation; generated tone is only a pipeline smoke input.
4. The package includes a single M2 face template and a single M3 speaker template; this is single-identity prototype scope.
5. The templates are sensitive biometric data. Use a **private deployment** and avoid public repository/Space visibility.
6. No account-registration/enrollment system, rate limiting, secure template lifecycle, or production security review is included.

## Conclusion
The UI, central policy flow, M1, M3 and M4 functional smoke checks work in the local test environment. M2 assets are packaged but real inference remains pending in an environment with its dependencies. The project is ready for a private prototype deployment attempt, followed by M2 runtime verification and real-input tests. It is not production-ready.
