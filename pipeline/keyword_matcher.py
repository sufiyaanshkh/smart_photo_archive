"""
Institutional Keyword & Taxonomy Matcher.

Matches image CLIP embeddings against institutional taxonomy embeddings with:
- Multi-prompt representations (keyword + visual cues + search expansions)
- Label-type aware routing (object, context, scene, metadata, entity_metadata)
- Sensitive context safeguarding (adheres to IIHS inference notes)
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .clip_embedder import CLIPEmbedder
from .institutional_taxonomy import InstitutionalTaxonomy, TaxonomyEntry

logger = logging.getLogger(__name__)


class KeywordMatcher:
    """Matches institutional taxonomy keywords to image embeddings."""

    def __init__(
        self,
        clip_embedder: Optional[CLIPEmbedder] = None,
        taxonomy: Optional[InstitutionalTaxonomy] = None
    ):
        self.clip_embedder = clip_embedder
        self.taxonomy = taxonomy
        self.keyword_list: List[str] = []
        self.taxonomy_records: List[Dict[str, Any]] = []
        self.keyword_embeddings: Optional[np.ndarray] = None

    def load_precomputed_embeddings(self, npy_path: str):
        """Loads precomputed keyword embeddings and metadata from a numpy file."""
        if not os.path.exists(npy_path):
            logger.warning(f"Keyword embeddings file not found: {npy_path}")
            return False

        try:
            data = np.load(npy_path, allow_pickle=True).item()
            self.keyword_embeddings = data.get('embeddings')
            self.keyword_list = data.get('keywords', [])
            self.taxonomy_records = data.get('records', [])
            logger.info(
                f"Loaded {len(self.keyword_list)} keyword embeddings from {npy_path} "
                f"(matrix shape: {self.keyword_embeddings.shape if self.keyword_embeddings is not None else None})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load keyword embeddings from {npy_path}: {e}")
            return False

    def build_taxonomy_embeddings(
        self,
        taxonomy: InstitutionalTaxonomy,
        clip_embedder: Optional[CLIPEmbedder] = None
    ) -> np.ndarray:
        """
        Builds multi-prompt averaged CLIP embeddings for all taxonomy entries.
        """
        embedder = clip_embedder or self.clip_embedder
        if embedder is None:
            raise ValueError("CLIPEmbedder is required to build embeddings.")

        self.taxonomy = taxonomy
        self.keyword_list = []
        self.taxonomy_records = []
        all_embeddings = []

        logger.info(f"Building multi-prompt embeddings for {len(taxonomy.entries)} taxonomy entries...")

        for entry in taxonomy.entries:
            prompts = entry.get_embedding_prompts()
            if not prompts:
                prompts = [entry.normalized_keyword]

            prompt_embs = embedder.embed_texts_batch(prompts)
            # Mean pool prompts and L2 normalize
            pooled = np.mean(prompt_embs, axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 1e-6:
                pooled = pooled / norm

            all_embeddings.append(pooled)
            self.keyword_list.append(entry.normalized_keyword)
            self.taxonomy_records.append({
                'id': entry.keyword_id,
                'keyword': entry.normalized_keyword,
                'original': entry.original_keyword,
                'category': entry.primary_category,
                'project': entry.project,
                'label_type': entry.label_type,
                'priority': entry.priority,
                'visual_class': entry.candidate_visual_class,
                'inference_note': entry.inference_note,
            })

        self.keyword_embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Generated taxonomy embeddings matrix: {self.keyword_embeddings.shape}")
        return self.keyword_embeddings

    def match_image_taxonomy(
        self,
        image_embedding: np.ndarray,
        top_n: int = 15,
        default_min_similarity: float = 0.20,
        type_thresholds: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Matches a single image embedding to institutional taxonomy.
        Applies label_type specific thresholds and sensitive attribute checks.
        """
        if self.keyword_embeddings is None or len(self.keyword_list) == 0:
            return []

        thresholds = type_thresholds or {
            'object': 0.22,
            'scene': 0.23,
            'context': 0.25,
            'entity_metadata': 0.28,
            'metadata': 0.30,
        }

        try:
            img_emb = image_embedding.reshape(1, -1)
            # Ensure normalized
            norm = np.linalg.norm(img_emb)
            if norm > 1e-6:
                img_emb = img_emb / norm

            similarities = np.dot(img_emb, self.keyword_embeddings.T).squeeze()
            top_indices = np.argsort(similarities)[::-1]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                rec = self.taxonomy_records[idx] if idx < len(self.taxonomy_records) else {}
                label_type = rec.get('label_type', 'context')
                threshold = thresholds.get(label_type, default_min_similarity)

                # Sensitive context guard: do not infer sensitive demographic/caste status purely on visual appearance
                inference_note = rec.get('inference_note', '')
                if inference_note and "do not infer sensitive status" in inference_note:
                    # Require significantly higher similarity to prevent false contextual tagging
                    threshold = max(threshold, 0.32)

                if score >= threshold:
                    match_method = f"clip_{label_type}"
                    results.append({
                        'keyword_id': rec.get('id', idx + 1),
                        'keyword': self.keyword_list[idx],
                        'score': score,
                        'category': rec.get('category', ''),
                        'project': rec.get('project', ''),
                        'label_type': label_type,
                        'priority': rec.get('priority', 'High'),
                        'match_method': match_method,
                    })

                if len(results) >= top_n:
                    break

            return results
        except Exception as e:
            logger.error(f"Taxonomy matching failed: {e}")
            return []

    def match_image(
        self,
        image_embedding: np.ndarray,
        top_n: int = 10,
        min_similarity: float = 0.2
    ) -> List[Tuple[str, float]]:
        """
        Backward-compatible matching returning (keyword, score) pairs.
        """
        matches = self.match_image_taxonomy(
            image_embedding, top_n=top_n, default_min_similarity=min_similarity
        )
        return [(m['keyword'], m['score']) for m in matches]

    def match_images_batch(
        self,
        image_embeddings: np.ndarray,
        top_n: int = 10,
        min_similarity: float = 0.2
    ) -> List[List[Tuple[str, float]]]:
        """
        Matches a batch of image embeddings to keywords.
        """
        return [
            self.match_image(image_embeddings[i], top_n=top_n, min_similarity=min_similarity)
            for i in range(image_embeddings.shape[0])
        ]
