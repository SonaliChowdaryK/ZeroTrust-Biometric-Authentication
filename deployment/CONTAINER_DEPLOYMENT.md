# Container deployment notes

This Docker setup is for the **placeholder UI only**. The biometric models and central policy evaluator are not included or connected.

## Build and run
From the repository root:
```bash
docker build -t zerotrust-biometric .
docker run --rm -p 7860:7860 zerotrust-biometric
```
Open `http://localhost:7860`.

## Deploy to a container host
Use the repository root as the build context and the included `Dockerfile`. Set the service port to `7860` (or the platform's injected `PORT`). Do not enable real authentication until all four model pipelines and frozen policy have been integrated, reviewed, and tested.

**Do not deploy or upload enrollment templates, personal face images, voice samples, secrets, or logs containing biometric information.**
