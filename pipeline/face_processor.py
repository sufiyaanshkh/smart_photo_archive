import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class FaceResult:
    bbox: Tuple[float, float, float, float]
    embedding: np.ndarray
    det_score: float
    age: int
    gender: str

class FaceProcessor:
    """InsightFace pipeline for face detection and recognition."""
    def __init__(self, model_name: str = 'buffalo_l', det_size: Tuple[int, int] = (640, 640), device: str = 'cuda'):
        self.det_size = det_size
        self.app = None
        logger.info(f"Loading InsightFace model {model_name}...")
        try:
            import insightface
            from insightface.app import FaceAnalysis
            providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0 if device=='cuda' else -1, det_size=det_size)
            logger.info("InsightFace loaded successfully.")
        except ImportError:
            logger.warning("InsightFace not installed. Face detection will be inactive until insightface is installed.")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace ({e}). Face detection will be inactive.")

    def process_image(self, image_path: str, min_det_score: float = 0.5) -> List[FaceResult]:
        """Processes a single image and returns a list of FaceResults."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image for face processing: {image_path}")
                return []
            
            faces = self.app.get(img)
            results = []
            
            for face in faces:
                if face.det_score < min_det_score:
                    continue
                # Ensure embedding is 512-d
                embedding = face.embedding
                if embedding is None:
                    continue
                    
                bbox = tuple(face.bbox)
                age = getattr(face, 'age', 0)
                gender = 'M' if getattr(face, 'gender', 0) == 1 else 'F'
                
                results.append(FaceResult(
                    bbox=bbox,
                    embedding=embedding,
                    det_score=float(face.det_score),
                    age=int(age),
                    gender=gender
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to process faces in {image_path}: {e}")
            return []

    def process_images_batch(self, image_paths: List[str], min_det_score: float = 0.5) -> Dict[str, List[FaceResult]]:
        """Processes a batch of images."""
        results = {}
        for path in image_paths:
            results[path] = self.process_image(path, min_det_score)
        return results
