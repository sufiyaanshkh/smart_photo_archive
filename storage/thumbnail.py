import os
import logging
from typing import List
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)

def generate_thumbnail(image_path: str, output_dir: str, size: int = 256) -> str:
    """
    Generates a thumbnail for a given image.
    Maintains aspect ratio and saves as JPEG with 85% quality.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        thumb_name = f"{name}_thumb.jpg"
        thumb_path = os.path.join(output_dir, thumb_name)
        
        if os.path.exists(thumb_path):
            return thumb_path
            
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g. RGBA or P)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
                
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            img.save(thumb_path, format="JPEG", quality=85)
            
        return thumb_path
    except Exception as e:
        logger.error(f"Failed to generate thumbnail for {image_path}: {e}")
        return ""

def generate_thumbnails_batch(image_paths: List[str], output_dir: str, size: int = 256, num_workers: int = 4) -> List[str]:
    """
    Generates thumbnails in batch using multiple workers.
    """
    results = []
    
    def process_func(path):
        return generate_thumbnail(path, output_dir, size)

    with ThreadPool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(process_func, image_paths), total=len(image_paths), desc="Generating thumbnails"):
            if res:
                results.append(res)
                
    return results
