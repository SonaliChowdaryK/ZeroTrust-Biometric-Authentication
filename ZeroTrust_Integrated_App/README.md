---
title: ZeroTrust Biometric Authentication
emoji: 🔐
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.11
suggested_hardware: cpu-basic
short_description: Four-module biometric authentication prototype
---

# ZeroTrust Biometric Authentication

A research/demo prototype that integrates four trained modules and the frozen central decision policy.

## Modules
- M1: Face Anti-Spoofing (official MiniFASNetV2 + MiniFASNetV1SE fusion)
- M2: Face Recognition (ArcFace w600k_r50 FP16 ONNX)
- M3: Speaker Recognition (SpeechBrain ECAPA-TDNN)
- M4: Voice Anti-Spoofing (official AASIST, ASVspoof 2019 LA checkpoint)

## Frozen decision policy
- Any structurally valid, completed module `REJECT` -> `DENY`.
- Invalid evidence -> `REQUIRE_ADDITIONAL_VERIFICATION`.
- Missing, duplicate, error, unavailable, not-run, or UNKNOWN module -> `REQUIRE_ADDITIONAL_VERIFICATION` (unless a valid completed rejection exists, which takes precedence).
- `ALLOW` only when M1, M2, M3, and M4 each appear exactly once, all complete successfully, and all return valid `ACCEPT`.

Thresholds retained:
- M2: 0.27845
- M3: 0.5235530659556389
- M4: 0.5

M1 uses the official Silent-Face-Anti-Spoofing two-checkpoint fusion: MiniFASNetV2 at scale 2.7 and MiniFASNetV1SE at scale 4.0. The official class mapping is output index 1 = LIVE; indices 0 and 2 = SPOOF. M1 uses the project's Widerface RetinaFace detector, expands the detected face box at each model's scale, resizes each crop to 80x80, and follows the original OpenCV BGR/float32 [0,255] tensor path without RGB swapping or normalization. The model and Caffe detector run on CPU. A missing face is reported as `unavailable`/`UNKNOWN`, not as a spoof rejection; malformed input or inference errors are reported as `error`/`UNKNOWN`.

The official source and pretrained assets are pinned to commit `b6d5f04ad78778917853b25c778acef6d5626d15` and SHA-256 verified by `scripts/download_m1_minifasnet.py`. The upstream repository is Apache-2.0 licensed; its license is retained at `models/M1/vendor/LICENSE`. The original repository requirements pin obsolete Torch/Torchvision versions, so they are not installed; this project uses its existing Torch, OpenCV, and NumPy dependencies.

The prior MobileNetV3 M1 evaluation record is not evidence for the replacement model. Strict checkpoint loading and CPU forward passes are checked locally, but genuine-user, printed-photo, and screen-replay experiments still need to be run with the steps below. No accuracy or anti-spoof guarantee is claimed.

## Local run
Recommended Python 3.11 environment.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_m1_minifasnet.py
python scripts/download_m4_aasist.py
python scripts/download_voice_challenge_model.py
python app.py
```

No new Python dependency was added for M1 or M4; AASIST uses the project's existing Torch and NumPy dependencies. The M4 downloader verifies the pinned official checkpoint by SHA-256 and retrieves it only if it is missing. Open the local URL printed by Gradio. First M2 initialization downloads the InsightFace `buffalo_l` detection assets if not cached. The project ArcFace recognition ONNX is included and is not re-downloaded.

## Interface

The compact light-theme authentication page pairs a face/voice capture card with a vertically stacked M1-M4 verification panel. Module statuses, decisions, scores, errors, and the central decision are driven by actual pipeline evidence; enrollment and registered-user management use the same visual theme. No authentication-history log is created.

## Enrollment and authentication

1. Open the **Enrollment** tab.
2. Enter a unique User ID and username.
3. Upload at least one usable face image and a voice sample of at least four seconds containing at least one second of speech.
4. Enroll the user. The system generates M2 and M3 templates and stores both under the same User ID.
5. Open **Authentication**, select the enrolled User ID, and provide new face and voice inputs.
6. M2 is compared only with that User ID's face template. M3 is compared only with that User ID's voice template.
7. M1 and M4 process the submitted face and voice independently. The existing central policy produces the final decision.

Use **User Management** to list enrolled identities or delete one identity and its associated templates. Re-enrollment is disabled by default and requires the explicit controlled update checkbox.

### Audio enhancement, VAD and model routing

- The uploaded/recorded source is left unchanged. Temporary mono WAV copies are made at 48 kHz for RNNoise and 16 kHz for model input, then deleted after processing.
- RNNoise performs actual recurrent-neural-network noise suppression on a separate copy. Silero VAD checks speech on both original and enhanced copies; silence, less than four seconds of recording, less than one second of detected speech, heavy clipping, and enhancement failures stop M3/M4 and request a new recording.
- M3 and M4 receive the unenhanced 16 kHz copy. M4 stays on original audio to preserve spoof cues. M3 also stays on original audio because controlled model comparisons showed denoising can change speaker scores inconsistently; RNNoise is used for audio-quality/VAD checks, not as a blind M3 preprocessor.
- A local Windows SAPI TTS experiment compared clean, fan-noise, traffic-noise, a second TTS voice in the background, and overlapping TTS voices. M3 similarity (original -> fully denoised) was: clean `0.8279 -> 0.8231`, fan `0.5919 -> 0.3958`, traffic `0.6510 -> 0.8287`, background voice `0.6254 -> 0.4124`, overlap `0.5192 -> 0.3559`. Partial RNNoise blends did not improve M3 over original audio across these cases.
- The previous MobileNetV3 M4 classified all these TTS samples as SPOOF. Those original -> denoised scores (clean `1.0000 -> 1.0000`, fan `0.9945 -> 0.9864`, traffic `0.9265 -> 0.9920`, background voice `1.0000 -> 1.0000`, overlap `1.0000 -> 1.0000`) are historical results for that retired M4 only; they say nothing about AASIST. They were synthetic-speech checks, not real replay-attack or human-speaker validation.
- RNNoise and VAD do not identify speaker count or reliably detect overlapping speech. The UI explicitly reports that overlap is not assessed; users must re-record alone in a quiet environment if other voices are audible. No speaker is isolated or selected automatically.
- RNNoise is provided by `pyrnnoise` (about 13 MB Windows wheel; CPU frame-based processing; Apache-2.0 binding over the BSD-3-Clause RNNoise implementation). Silero VAD is packaged locally by `silero-vad` (MIT; CPU inference). The first run does not require downloading VAD weights separately.
- Authentication also presents a fresh four-word phrase that expires after three minutes. A local Vosk English recognizer checks that phrase against the original mono 16 kHz recording, not the RNNoise copy, before M3/M4 results can be considered for authorization; voice audio is not sent to an external speech service. If a valid recording fails phrase recognition, M3/M4 may run in diagnostic-only mode. Their separate results are displayed but excluded from the authoritative evidence and final decision. Invalid or unsuitable audio still prevents voice inference. Install `vosk` from `requirements.txt` and run `python scripts/download_voice_challenge_model.py` once to download the approximately 40 MB Apache-2.0 model. It uses about 300 MB of memory at runtime.
- The phrase challenge blocks reuse of a fixed older recording, but it is not proof of liveness and cannot stop a replay or generated voice that speaks the current phrase. Diagnostic M3/M4 results never override a failed phrase check; M4 remains mandatory for an authoritative attempt. The app clears the used audio clip after a challenge failure/expiry or completed attempt; record the newly displayed phrase in a fresh clip. Validate it with genuine human replay tests before making anti-replay security claims. Uploaded audio must contain the currently displayed phrase too.

### M1 independent camera and attack experiments (Windows)

From the project directory, verify/download the pinned M1 files and launch the independent test tool:

```powershell
py -3.11 scripts\download_m1_minifasnet.py
py -3.11 scripts\m1_minifasnet_test_tool.py --webcam --samples 30
```

The webcam opens only when `--webcam` is specified. Position yourself in front of the camera, press **ENTER** to test a frame, or type **Q** and press ENTER to exit. This console capture works with the project's `opencv-python-headless` dependency and does not open a preview window. The two classifiers and RetinaFace detector are loaded once, then reused. Each sample logs face detection, model decisions, actual probabilities and raw logits, bounding box, and processing time. Copy the terminal output to a log if you want to keep an experiment record.

Run Experiment A with one person in frame, holding still and keeping the whole face visible. Capture multiple frames by pressing ENTER for each sample in these conditions: normal room lighting, bright lighting, low lighting, and near/far camera distances. Run the command separately for each condition if you want to label the terminal output unambiguously.

For Experiment B, display that person's photograph on a phone and point the webcam at the phone. For Experiment C, replay a face video or image on another screen and test multiple frames. These are local demonstrations, not comprehensive presentation-attack evaluation. MiniFASNet's upstream README cautions that camera and scene differences affect performance and that inputs should be camera-captured.

To test uploaded images instead of opening the webcam:

```powershell
py -3.11 scripts\m1_minifasnet_test_tool.py --images C:\path\to\live_sample.jpg C:\path\to\phone_photo.jpg C:\path\to\screen_replay.png
```

Record the actual JSON-like output per case; do not infer general accuracy from one or two samples. This repository does not currently include genuine-person or attack images, so those experiment results remain unmeasured until you run them.

### M4 AASIST voice anti-spoofing test (Windows)

M4 now uses the official AASIST architecture and `models/M4/aasist_2019_la.pth` from the official [clovaai/aasist](https://github.com/clovaai/aasist) repository, pinned to commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`. The checkpoint is 1,281,532 bytes with SHA-256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`. The upstream architecture, AASIST config, and MIT license are retained in `models/M4/vendor/`. Strict checkpoint loading and CPU inference were verified locally. The upstream CLI itself exits when no GPU is found, but the upstream `Model` class and checkpoint were separately strict-loaded and forward-tested on CPU; the app uses that model class directly.

The official config is track `LA` and sets `nb_samp=64600`; AASIST's first convolution is configured for 16 kHz, so the model input is mono 16 kHz waveform data. Upstream `data_utils.py` maps `bonafide` to class 1 and spoof to class 0; upstream evaluation uses output column 1 as the bona fide score. Model output values are logits, not probabilities. This integration retains the raw logits and official class scores; displayed softmax scores are derived, explicitly uncalibrated, and are not guarantees of confidence. The existing M4 threshold remains `0.5` on the two-class softmax output (equivalent to selecting the larger logit); it has not been independently calibrated for this application.

The official repository describes ASVspoof 2019 as its training/evaluation dataset and reports 0.83% EER and 0.0275 min t-DCF for AASIST on that benchmark. This checkpoint is for the **logical-access** track (spoofed/synthesized/converted speech), not the physical-access phone-speaker replay track. Its published benchmark does not establish accuracy on your microphone, voice, or phone-speaker replays. No genuine-person or replay experiment has been run here.

M4 keeps receiving the original converted audio in authentication. The standalone probe can optionally run a second, separate inference on the RNNoise copy; that diagnostic does not change application routing. The official input length is 64,600 samples (4.0375 s at 16 kHz). Inputs shorter than four seconds are rejected; valid 4.0–4.0375 s recordings are repeat-padded only to reach the official sample count. Longer recordings use the first 64,600 samples. Source amplitude is preserved; no peak normalization or Mel features are applied.

Verify or download the pinned checkpoint, then test one or more saved recordings. The probe loads the model once, reports input/decode metadata, raw logits, official scores, derived softmax scores, prediction, timing, and errors:

```powershell
py -3.11 scripts\download_m4_aasist.py
py -3.11 scripts\m4_audio_probe.py C:\path\to\genuine.wav C:\path\to\replay.wav
```

To run a second, separate inference on the project's RNNoise-enhanced copy while still testing the original first:

```powershell
py -3.11 scripts\m4_audio_probe.py --compare-denoised C:\path\to\genuine.wav C:\path\to\replay.wav
```

The comparison is diagnostic only; it does not change authentication routing. For genuine tests, record several fresh sentences in a quiet room and with modest background noise. For replay tests, first record your own speech, then play it through a phone speaker and record the playback at varied volumes/distances. Keep each sample labeled and compare the JSON outputs. The challenge phrase must be current and match before any M3/M4 result is authoritative; a failed phrase check makes voice results diagnostic-only. Do not treat synthetic audio or a small number of samples as model validation.

### Microphone troubleshooting

The audio player should show a non-zero duration and a "Voice sample received" status before enrollment or verification. If microphone recording remains at `0:00` in the VS Code embedded preview, open the same local app URL in Chrome or Edge, or use **Upload file** instead. A missing/zero-duration sample has not reached M3 or M4; it is not a biometric rejection. For authentication, say the displayed four-word challenge in order and record at least four seconds with only one audible speaker.

## Storage and privacy

- Raw face images and audio recordings are not intentionally saved by the enrollment workflow.
- Face templates are stored as 512-dimensional `.npy` arrays.
- Voice templates are stored as 192-dimensional PyTorch tensors.
- Records are stored below `enrollments/<safe-user-id>/`.
- The directory and template files receive restrictive permissions where the operating system supports it.
- This is a prototype; it does not provide encrypted-at-rest storage, hardware-backed key protection, or production account security.
- The `enrollments/` directory is excluded by `.gitignore`.

## Deployment (Hugging Face Spaces)
1. Create a **Private Gradio Space** (strongly recommended because this package contains personal face/speaker enrollment templates).
2. Do not publish the package, enrollment templates, or checkpoints in a public GitHub repository or public Space.
3. Optionally set Space secrets `ZT_APP_USERNAME` and `ZT_APP_PASSWORD` to enable Gradio basic login; set both or neither. Keep the Space private regardless.
4. Upload the project files and model directories, or clone the Space repository and copy this project into it.
5. Run `python scripts/download_voice_challenge_model.py` during setup so the local phrase recognizer is available.
6. Ensure large model files are stored using Git LFS.
7. Use the `README.md` Space metadata at the top of this file.
8. Set CPU hardware for a prototype. Model startup and inference may be slow on CPU.
9. Open the Space and test with genuine and spoof samples. Do not use it for real-world authentication until the biometric models and thresholds have been independently evaluated.

## Important limitations
- Packaged M2/M3 demo templates remain in `models/` for compatibility and fallback asset checks, but authentication requires an explicitly enrolled User ID and never silently falls back to those demo templates.
- The prior M1 MobileNetV3 validation report showed BPCER approximately 59.78%; that result does not apply to the MiniFASNet replacement. The replacement has not yet been evaluated with genuine faces or presentation attacks.
- M2 and M3 preliminary test reports use very small sample counts and thresholds are not independently calibrated.
- `models/M4/evaluation_metrics.json` and `best_voice_antispoof.pth` are retained legacy MobileNetV3 artifacts; they are not used by AASIST and their metrics do not describe the replacement.
- AASIST's official weights are trained/evaluated on ASVspoof 2019 LA. Physical phone-speaker replay performance, genuine-user false rejection, and denoising effects have not been validated with human samples in this project.
- M3 is speaker verification, not liveness; M4 is responsible for anti-spoofing.
- This is a prototype, not production-grade security. Biometric templates are sensitive data. Keep deployment private and protect authentication, logs, and uploads.

## Project layout
- `app.py` — Gradio interface
- `integration/` — adapters, pipeline, evidence validation, policy, central engine and decision output
- `models/M1` ... `models/M4` — model sources/checkpoints/templates
- `tests/` — policy and contract tests
