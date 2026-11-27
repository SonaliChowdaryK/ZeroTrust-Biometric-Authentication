#!/usr/bin/env python3
"""Synthetic functional smoke tests; these are NOT biometric accuracy tests."""
from pathlib import Path
import argparse, sys, tempfile, wave
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

def make_tone(path):
    sr=16000
    t=np.arange(sr*4)/sr
    y=(0.2*np.sin(2*np.pi*220*t)).astype(np.float32)
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((y*32767).astype(np.int16).tobytes())

def load_file_mod(name, path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def main(include_m2=False):
    print("=== ZeroTrust synthetic model smoke tests ===")
    # M1: official checkpoint/detector load + synthetic non-face input.
    from PIL import Image
    from biometric_adapters import M1FaceAntiSpoofingAdapter
    m1=M1FaceAntiSpoofingAdapter()
    with tempfile.TemporaryDirectory() as td:
        image_path=Path(td)/"noise_image.png"
        Image.fromarray(np.random.default_rng(42).integers(0,255,(320,320,3),dtype=np.uint8)).save(image_path)
        m1_result=m1.verify(str(image_path))
        print(
            "M1 MiniFASNetV2 fusion/detector runtime: PASS; synthetic non-face status:",
            m1_result["status"],
            m1_result.get("decision"),
        )

        # M3/M4 use generated tone to verify the runtime can execute; not accuracy evidence.
        audio_path=Path(td)/"synthetic_tone.wav"; make_tone(audio_path)
        m3mod=load_file_mod("smoke_m3",ROOT/"models/M3/m3_speaker_recognition.py")
        m3=m3mod.M3SpeakerRecognition(model_source=str(ROOT/"models/M3/pretrained_models/spkrec-ecapa-voxceleb"),
          savedir=str(ROOT/"models/M3/pretrained_models/spkrec-ecapa-voxceleb"),threshold=0.5235530659556389,device="cpu")
        m3.load_enrollment(str(ROOT/"models/M3/M3_voice_enrollment_template.pt"))
        m3_result=m3.verify(str(audio_path))
        print("M3 local checkpoints/inference: PASS; synthetic-tone decision:",m3_result["decision"])

        m4mod=load_file_mod("smoke_m4",ROOT/"models/M4/m4_voice_antispoofing.py")
        m4=m4mod.M4VoiceAntiSpoofing(str(ROOT/"models/M4/aasist_2019_la.pth"),threshold=0.5)
        m4_result=m4.predict(str(audio_path))
        print("M4 checkpoint/inference: PASS; synthetic-tone decision:",m4_result["decision"])

    # M2 assets (model and template) are checked independently.
    m2_model=ROOT/"models/M2/arcface_w600k_r50_fp16.onnx"
    m2_template=ROOT/"models/M2/M2_face_enrollment_template.npy"
    assert m2_model.exists() and m2_template.exists()
    template=np.load(m2_template)
    assert template.shape==(512,)
    print("M2 assets/template: PASS; model bytes:",m2_model.stat().st_size, "template:",template.shape)
    if include_m2:
        from biometric_adapters import M2FaceRecognitionAdapter
        adapter=M2FaceRecognitionAdapter()
        with tempfile.TemporaryDirectory() as td:
            blank=Path(td)/"blank.png"
            Image.new("RGB",(320,320),(128,128,128)).save(blank)
            print("M2 blank-image smoke result:",adapter.verify(str(blank)))
    print("Smoke tests complete. Synthetic results are not accuracy evidence.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--m2",action="store_true",help="Initialize InsightFace detector and ArcFace runtime; may download buffalo_l detector assets.")
    args=ap.parse_args()
    main(args.m2)
