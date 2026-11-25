# Deployment scaffold

This folder contains a **placeholder UI**, not the integrated biometric system.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

The app intentionally reports all four modules as NOT CONNECTED and returns a safe placeholder decision. Do not use it for authentication.

## Next integration steps
1. Transfer the complete project source and model checkpoints through a secure upload / repository workflow.
2. Reconcile the source modules with the existing frozen central policy evaluator and adapters.
3. Keep biometric enrollment templates private and out of Git.
4. Test each module with valid, invalid, spoof, missing, and error cases.
5. Run end-to-end policy tests; only enable real authentication after verification.
6. Choose a deployment platform with sufficient memory / CPU / GPU and review biometric privacy controls.
