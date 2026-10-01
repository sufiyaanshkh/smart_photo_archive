import logging
import torch
from typing import List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class ImageTagger:
    """
    RAM++ open-vocabulary tagger.
    """
    def __init__(self, model_name: str = 'xinyu1205/recognize-anything-plus-plus', device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading RAM++ model on {self.device}...")
        try:
            try:
                from ram.models import ram_plus
                from ram import inference_ram_openset as inference
                from ram import get_transform
                self.model = ram_plus(pretrained=model_name, image_size=384, vit='swin_l')
                self.model.eval().to(self.device)
                self.transform = get_transform(image_size=384)
                self.use_ram = True
            except ImportError:
                logger.warning("RAM package missing. Using HuggingFace CLIP zero-shot fallback.")
                from transformers import pipeline
                self.pipe = pipeline("image-classification", model="openai/clip-vit-large-patch14", device=0 if self.device=='cuda' else -1)
                self.use_ram = False
            logger.info("Tagger model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tagger model: {e}")
            raise

    def tag_image(self, image_path: str, min_confidence: float = 0.5) -> List[Tuple[str, float]]:
        """Tags a single image."""
        try:
            if getattr(self, 'use_ram', False):
                from ram import inference_ram_openset as inference
                image = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
                res = inference(image, self.model)
                tags = res[0].split(' | ')
                return [(tag.strip(), 1.0) for tag in tags if tag.strip()]
            else:
                image = Image.open(image_path).convert('RGB')
                preds = self.pipe(image, candidate_labels=["indoor", "outdoor", "person", "nature", "animal", "car", "building"])
                return [(p['label'], p['score']) for p in preds if p['score'] >= min_confidence]
        except Exception as e:
            logger.error(f"Failed to tag image {image_path}: {e}")
            return []

    def tag_images_batch(self, image_paths: List[str], min_confidence: float = 0.5) -> List[List[Tuple[str, float]]]:
        """Tags a batch of images."""
        return [self.tag_image(path, min_confidence) for path in image_paths]
