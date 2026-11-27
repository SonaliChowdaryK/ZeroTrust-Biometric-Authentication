
import os
import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.inference.speaker import SpeakerRecognition


class M3SpeakerRecognition:

    def __init__(
        self,
        model_source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        threshold=0.5235530659556389,
        device=None
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold

        self.model = SpeakerRecognition.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": self.device}
        )

        self.enrollment_template = None

    # --------------------------------
    # Audio preprocessing
    # --------------------------------
    def preprocess_audio(self, audio_path):

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.numel() == 0:
            raise ValueError("Audio file is empty.")

        # Convert multichannel audio to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16000
            )

        duration = waveform.shape[1] / 16000

        if duration < 1.0:
            raise ValueError(
                "Audio is too short. Please provide at least 1 second."
            )

        return waveform.to(self.device)

    # --------------------------------
    # Extract speaker embedding
    # --------------------------------
    def extract_embedding(self, audio_path):

        waveform = self.preprocess_audio(audio_path)

        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)

        embedding = embedding.squeeze()
        embedding = F.normalize(embedding, p=2, dim=0)

        return embedding

    # --------------------------------
    # Enroll speaker
    # --------------------------------
    def enroll(self, audio_paths):

        if not audio_paths:
            raise ValueError("Provide at least one enrollment audio file.")

        embeddings = [
            self.extract_embedding(path)
            for path in audio_paths
        ]

        template = torch.stack(embeddings).mean(dim=0)
        template = F.normalize(template, p=2, dim=0)

        self.enrollment_template = template

        return template

    # --------------------------------
    # Save enrollment template
    # --------------------------------
    def save_enrollment(self, file_path):

        if self.enrollment_template is None:
            raise ValueError("No enrollment template to save.")

        torch.save(
            self.enrollment_template.detach().cpu(),
            file_path
        )

        print(f"Enrollment template saved: {file_path}")

    # --------------------------------
    # Load enrollment template
    # --------------------------------
    def load_enrollment(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        template = torch.load(
            file_path,
            map_location=self.device,
            weights_only=True
        )

        if template.shape != (192,):
            raise ValueError("Invalid enrollment template dimension.")

        self.enrollment_template = F.normalize(
            template.to(self.device),
            p=2,
            dim=0
        )

        print("Enrollment template loaded successfully.")

    # --------------------------------
    # Verify speaker
    # --------------------------------
    def verify(self, audio_path, threshold=None):

        if self.enrollment_template is None:
            raise ValueError(
                "No speaker enrolled. Call enroll() or load_enrollment() first."
            )

        threshold = (
            self.threshold if threshold is None else threshold
        )

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        test_embedding = self.extract_embedding(audio_path)

        similarity = F.cosine_similarity(
            self.enrollment_template.unsqueeze(0),
            test_embedding.unsqueeze(0)
        ).item()

        decision = (
            "ACCEPT"
            if similarity >= threshold
            else "REJECT"
        )

        return {
            "module_id": "M3",
            "module_name": "Speaker Recognition",
            "status": "completed",
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "decision": decision,
            "verified": decision == "ACCEPT",
            "model": "SpeechBrain ECAPA-TDNN",
            "embedding_dimension": 192
        }
