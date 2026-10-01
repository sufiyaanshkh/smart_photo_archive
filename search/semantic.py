"""Semantic search using CLIP embeddings."""

import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class SemanticSearch:
    """Semantic search engine using CLIP embeddings."""
    
    def __init__(self, clip_embedder: Any, vector_store: Any) -> None:
        """
        Initialize the semantic search engine.
        
        Args:
            clip_embedder: The CLIP embedder instance.
            vector_store: The FAISS vector store instance.
        """
        self.clip_embedder = clip_embedder
        self.vector_store = vector_store
        
    def search(self, query_text: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search for images using a text query.
        
        Args:
            query_text: The text query.
            top_k: Number of results to return.
            
        Returns:
            List of (image_id, score) tuples.
        """
        try:
            if hasattr(self.clip_embedder, 'embed_text'):
                query_embedding = self.clip_embedder.embed_text(query_text)
            else:
                query_embedding = self.clip_embedder.encode_text(query_text)
            results = self.vector_store.search(query_embedding, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Semantic text search failed for query '{query_text}': {e}")
            return []
            
    def search_by_image(self, image_path: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search for images similar to the provided image.
        
        Args:
            image_path: Path to the query image.
            top_k: Number of results to return.
            
        Returns:
            List of (image_id, score) tuples.
        """
        try:
            if hasattr(self.clip_embedder, 'embed_image'):
                query_embedding = self.clip_embedder.embed_image(image_path)
            else:
                query_embedding = self.clip_embedder.encode_image(image_path)
            results = self.vector_store.search(query_embedding, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Semantic image search failed for image '{image_path}': {e}")
            return []
