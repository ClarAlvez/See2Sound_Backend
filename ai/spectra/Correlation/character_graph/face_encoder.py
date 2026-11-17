"""Optional OpenCV Zoo YuNet + SFace identity embeddings for person crops.

No download or network access at import time. Fail closed without the ONNX
weights: absent/ambiguous faces do not produce an embedding or a fake ID.
"""
from __future__ import annotations
from pathlib import Path
import math


class SFaceEncoder:
    def __init__(self, detector_model, recognition_model, *, detection_threshold=0.87):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        detector_model, recognition_model = Path(detector_model), Path(recognition_model)
        if not (detector_model.is_file() and recognition_model.is_file()):
            raise FileNotFoundError('Modelos YuNet e SFace devem estar presentes em data/models/Face/.')
        if not hasattr(cv2, 'FaceDetectorYN') or not hasattr(cv2, 'FaceRecognizerSF'):
            raise RuntimeError('OpenCV sem FaceDetectorYN/FaceRecognizerSF (use opencv-python 4.13).')
        self.detector = cv2.FaceDetectorYN.create(str(detector_model), '', (320, 320),
                                                 float(detection_threshold), 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognition_model), '')

    def encode(self, image_path):
        cv2, np = self.cv2, self.np
        path = Path(image_path)
        if not path.is_file():
            return None
        try:
            # imdecode supports non-ASCII Windows paths.
            pixels = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
            if image is None or min(image.shape[:2]) < 30:
                return None
            h, w = image.shape[:2]
            # Avoid full-resolution face detection on large full-body crops.
            scale = min(1.0, 640.0 / max(w, h))
            if scale < 1.0:
                image = cv2.resize(image, (max(32, round(w * scale)), max(32, round(h * scale))))
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) != 1:
                # Multiple faces in a crop are ambiguous. Never pick at random.
                return None
            face = faces[0]
            if float(face[2]) < 26 or float(face[3]) < 26:
                return None
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).reshape(-1).astype('float32')
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm < 1e-10:
                return None
            return (vector / norm).tolist()
        except (OSError, ValueError, cv2.error):
            return None
