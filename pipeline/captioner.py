import logging
import torch
from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ImageCaptioner:
    """Florence-2 image captioner."""
    def __init__(self, model_name: str = 'microsoft/Florence-2-large', device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Florence-2 model {model_name} on {self.device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval().to(self.device)
            logger.info("Florence-2 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            raise

    def caption_image(self, image_path: str) -> str:
        """Generates a caption for a single image."""
        return self.caption_images_batch([image_path], batch_size=1)[0]

    def caption_images_batch(self, image_paths: List[str], batch_size: int = 16) -> List[str]:
        """Generates captions for a batch of images."""
        captions = []
        task_prompt = '<MORE_DETAILED_CAPTION>'
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Could not open {path}: {e}")
            
            if not images:
                captions.extend([""] * len(batch_paths))
                continue

            try:
                inputs = self.processor(text=[task_prompt]*len(images), images=images, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )
                    
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
                
                batch_captions = [""] * len(batch_paths)
                for j, text in enumerate(generated_texts):
                    parsed_answer = self.processor.post_process_generation(text, task=task_prompt, image_size=(images[j].width, images[j].height))
                    batch_captions[valid_indices[j]] = parsed_answer.get(task_prompt, text)
                    
                captions.extend(batch_captions)
                
            except Exception as e:
                logger.error(f"Batch caption generation failed: {e}")
                captions.extend([""] * len(batch_paths))
                
        return captions
