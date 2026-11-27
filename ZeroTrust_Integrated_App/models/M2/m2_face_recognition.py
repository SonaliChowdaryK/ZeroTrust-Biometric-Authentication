
import cv2
import numpy as np
from insightface.utils import face_align


class FaceRecognitionM2:

    def __init__(self, face_app, recognition_session, threshold=0.27845):
        self.face_app = face_app
        self.session = recognition_session
        self.threshold = threshold
        self.template = None

    def get_aligned_face(self, image_path):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.face_app.get(image)

        valid_faces = [
            face for face in faces
            if (face.bbox[2] - face.bbox[0]) >= 80
            and (face.bbox[3] - face.bbox[1]) >= 80
        ]

        if not valid_faces:
            return None, "No valid face detected"

        face = max(
            valid_faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        aligned = face_align.norm_crop(
            image,
            landmark=face.kps,
            image_size=112
        )

        return aligned, "Face detected and aligned"

    def get_embedding(self, aligned_face):
        blob = cv2.dnn.blobFromImages(
            [aligned_face],
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True
        ).astype(np.float16)

        embedding = self.session.run(
            None,
            {"input.1": blob}
        )[0].flatten().astype(np.float32)

        embedding /= (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def enroll(self, image_paths):
        embeddings = []

        for image_path in image_paths:
            aligned, status = self.get_aligned_face(image_path)

            if aligned is None:
                print(f"Skipped {image_path}: {status}")
                continue

            embeddings.append(self.get_embedding(aligned))

        if not embeddings:
            raise ValueError("Enrollment failed: no valid faces found")

        embeddings = np.array(embeddings)

        template = np.mean(embeddings, axis=0)
        template /= (np.linalg.norm(template) + 1e-10)

        self.template = template

        return {
            "status": "enrolled",
            "images_used": len(embeddings),
            "template_dimension": len(template)
        }

    def verify(self, image_path):
        if self.template is None:
            raise RuntimeError("No identity enrolled. Call enroll() first.")

        aligned, status = self.get_aligned_face(image_path)

        if aligned is None:
            return {
                "status": "failed",
                "reason": status,
                "verified": False,
                "similarity": None,
                "decision": "REJECT"
            }

        embedding = self.get_embedding(aligned)

        similarity = float(np.dot(self.template, embedding))
        verified = similarity >= self.threshold

        return {
            "status": "success",
            "similarity": similarity,
            "threshold": self.threshold,
            "verified": verified,
            "decision": "ACCEPT" if verified else "REJECT"
        }


    def verify_with_evidence(self, image_path):
        result = self.verify(image_path)

        return {
            "module_id": "M2",
            "module_name": "Face Recognition",
            "status": result["status"],
            "verified": result["verified"],
            "decision": result["decision"],
            "similarity": result.get("similarity"),
            "threshold": result.get("threshold", self.threshold),
            "model": "ArcFace w600k_r50 FP16",
            "embedding_dimension": 512
        }
