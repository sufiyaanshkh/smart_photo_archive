import logging
import numpy as np
from typing import List, Union
from PIL import Image

logger = logging.getLogger(__name__)

class CLIPEmbedder:
    """
    CLIP embedding pipeline.
    Supports open_clip as primary, HuggingFace transformers as fallback,
    and deterministic fallback vectors if torch/open_clip are not installed.
    """
    def __init__(self, model_name: str = 'ViT-L-14', pretrained: str = 'openai', device: str = 'cuda'):
        self.backend = None
        self.device = device
        
        # 1. Try open_clip
        try:
            import torch
            import open_clip
            self.device = device if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading open_clip model {model_name} on {self.device}...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            self.backend = 'open_clip'
            self.torch = torch
            logger.info("open_clip model loaded successfully.")
            return
        except ImportError:
            logger.warning("open_clip not found. Trying HuggingFace transformers fallback...")
        except Exception as e:
            logger.warning(f"Failed to load open_clip model ({e}). Trying transformers fallback...")

        # 2. Try transformers
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self.device = device if torch.cuda.is_available() else 'cpu'
            hf_name = "openai/clip-vit-large-patch14" if "L" in model_name else "openai/clip-vit-base-patch32"
            logger.info(f"Loading HuggingFace CLIP model {hf_name} on {self.device}...")
            self.hf_model = CLIPModel.from_pretrained(hf_name).to(self.device)
            self.hf_processor = CLIPProcessor.from_pretrained(hf_name)
            self.hf_model.eval()
            self.backend = 'transformers'
            self.torch = torch
            logger.info("HuggingFace CLIP loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"transformers CLIP could not be loaded: {e}. Using deterministic fallback embedder.")

        # 3. Fallback mock
        self.backend = 'mock'
        self.dim = 768

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embeds a single image and returns L2-normalized numpy array."""
        if self.backend == 'open_clip':
            try:
                image = Image.open(image_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with self.torch.no_grad():
                    features = self.model.encode_image(image_input)
                    features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().squeeze()
            except Exception as e:
                logger.error(f"Failed to embed image {image_path}: {e}")
                return np.zeros(768, dtype=np.float32)
        elif self.backend == 'transformers':
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = self.hf_processor(images=image, return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    features = self.hf_model.get_image_features(**inputs)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu().numpy().squeeze()
            except Exception as e:
                logger.error(f"Failed to embed image {image_path}: {e}")
                return np.zeros(768, dtype=np.float32)
        else:
            # Deterministic pseudo-embedding based on filename hash
            h = hash(image_path) % (2**31)
            rng = np.random.RandomState(h)
            v = rng.randn(self.dim).astype(np.float32)
            return v / np.linalg.norm(v)

    def embed_images_batch(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Embeds a batch of images and returns L2-normalized numpy array."""
        if not image_paths:
            return np.empty((0, 768), dtype=np.float32)
        all_features = [self.embed_image(p) for p in image_paths]
        return np.vstack(all_features)

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string."""
        return self.embed_texts_batch([text]).squeeze()

    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of text strings."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        if self.backend == 'open_clip':
            try:
                text_input = self.tokenizer(texts).to(self.device)
                with self.torch.no_grad():
                    features = self.model.encode_text(text_input)
                    features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()
            except Exception as e:
                logger.error(f"Failed to embed texts with open_clip: {e}")
                return np.zeros((len(texts), 768), dtype=np.float32)
        elif self.backend == 'transformers':
            try:
                inputs = self.hf_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with self.torch.no_grad():
                    features = self.hf_model.get_text_features(**inputs)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu().numpy()
            except Exception as e:
                logger.error(f"Failed to embed texts with transformers: {e}")
                return np.zeros((len(texts), 768), dtype=np.float32)
        else:
            vectors = []
            for t in texts:
                h = abs(hash(t)) % (2**31)
                rng = np.random.RandomState(h)
                v = rng.randn(self.dim).astype(np.float32)
                vectors.append(v / np.linalg.norm(v))
            return np.vstack(vectors)
