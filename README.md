# ZeroTrust Biometric Authentication

A four-module biometric verification system (M1 face anti-spoofing, M2 face recognition, M3 speaker recognition, M4 voice anti-spoofing) using a frozen central policy evaluator.

## Status
This repository is being prepared for remote development and deployment. The initial deployment scaffold is **not yet the complete biometric application**. Model code, checkpoints, enrollment templates, and integration policy must be transferred and tested before real authentication is enabled.

## Privacy / security
- Never commit personal face images, voice recordings, enrollment templates, secrets, or logs containing biometric data.
- Enrollment templates are sensitive biometric data. Store them securely and do not publish them.
- Authentication must fail closed: missing, invalid, incomplete, or unavailable module evidence must not result in ALLOW.
- M1 quality limitations and the frozen decision thresholds must be disclosed and preserved.

## Deployment
The current scaffold is a UI shell only. See `deployment/README.md`.
