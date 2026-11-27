# Deployment Checklist (Hugging Face Spaces)

## Before uploading
- [ ] Use a **Private** Space. The package includes personal M2 face and M3 speaker enrollment templates.
- [ ] Do not publish the project ZIP, templates, or checkpoints in a public repository.
- [ ] Decide whether to set `ZT_APP_USERNAME` and `ZT_APP_PASSWORD` as Space secrets. If set, set both.
- [ ] Confirm you are comfortable hosting your biometric templates on the selected provider.

## Create the Space
1. Sign in to Hugging Face.
2. Create a new Space.
3. Select **Gradio** SDK.
4. Select **Private** visibility.
5. Choose CPU hardware for the prototype.
6. Upload the contents of `ZeroTrust_Integrated_App` to the Space repository.
7. Keep large checkpoint files under Git LFS (Hugging Face Spaces handles large files with LFS).
8. Commit/push. The Space will build from `requirements.txt`.

## Optional login
In Space Settings -> Variables and Secrets, add:
- `ZT_APP_USERNAME` (secret)
- `ZT_APP_PASSWORD` (secret)

If only one is set, the app stops at startup. If neither is set, the app has no app-level login; use a private Space.

## First-run expectations
- CPU package installation can take time.
- The first M2 request may download InsightFace `buffalo_l` detector assets.
- M3/M4 CPU inference can be slow.
- The app requires both an image and audio input for a complete authentication attempt.

## Test after deployment
- [ ] Face upload/webcam component works.
- [ ] Audio upload/microphone component works.
- [ ] M1 runs.
- [ ] M2 initializes detector and ArcFace runtime.
- [ ] M3 runs with local ECAPA checkpoints/template.
- [ ] M4 runs with `.pth` checkpoint and threshold 0.5.
- [ ] All-accept policy test returns ALLOW.
- [ ] Valid reject policy test returns DENY.
- [ ] Missing/error/unknown evidence returns REQUIRE_ADDITIONAL_VERIFICATION unless valid reject precedence applies.
- [ ] Test with representative genuine and spoof/replay/synthetic audio/face inputs before treating the prototype as trustworthy.

## Security note
This prototype is not a production authentication system. The current package contains a single M2 face template and a single M3 speaker template. It does not implement multi-user registration, secure enrollment, template revocation, rate limiting, account recovery, production audit storage, or a threat-tested web authentication boundary. Keep it private and do not use it to protect valuable accounts or resources.
