"""
Hybrid search engine with taxonomy-aware query expansion and filtering.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Set

from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    image_id: int
    file_path: str
    thumbnail_path: str
    score: float
    matched_tags: Optional[List[str]] = field(default_factory=list)
    caption: Optional[str] = None
    faces: Optional[List[Any]] = field(default_factory=list)
    institutional_tags: Optional[List[Dict[str, Any]]] = field(default_factory=list)


class SearchEngine:
    """Hybrid search engine that combines semantic, keyword, face search and institutional taxonomy."""

    def __init__(
        self,
        semantic_search: Any,
        keyword_search: Any,
        face_search: Any,
        fusion_method: str = 'rrf'
    ) -> None:
        self.semantic_search = semantic_search
        self.keyword_search = keyword_search
        self.face_search = face_search
        self.fusion_method = fusion_method

    def _expand_query_with_taxonomy(self, query: str) -> str:
        """
        Expands user search query if it matches taxonomy terms or categories.
        """
        try:
            db = self.keyword_search.db
            q_lower = query.lower().strip()
            with db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT search_expansions, visual_cues
                    FROM institutional_keywords
                    WHERE LOWER(normalized_keyword) = ?
                       OR LOWER(original_keyword) = ?
                       OR LOWER(primary_category) = ?
                    LIMIT 2
                ''', (q_lower, q_lower, q_lower))
                rows = cursor.fetchall()

            expansions: Set[str] = set()
            for r in rows:
                if r['search_expansions']:
                    for term in r['search_expansions'].split(';'):
                        t = term.strip()
                        if t and t.lower() != q_lower:
                            expansions.add(t)

            if expansions:
                # Add up to 3 expansion keywords
                added_terms = " ".join(list(expansions)[:3])
                expanded = f"{query} {added_terms}"
                logger.info(f"Expanded query '{query}' -> '{expanded}'")
                return expanded
        except Exception as e:
            logger.debug(f"Taxonomy query expansion skipped: {e}")

        return query

    def search(
        self,
        query: str,
        top_k: int = 50,
        mode: str = 'hybrid',
        category: Optional[str] = None,
        project: Optional[str] = None,
        label_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Execute a search based on text query with optional institutional filters.
        """
        raw_results: List[Tuple[int, float]] = []

        # Taxonomy expansion for semantic queries
        expanded_query = self._expand_query_with_taxonomy(query)

        try:
            if mode == 'hybrid':
                semantic_res = self.semantic_search.search(expanded_query, top_k=top_k * 2)
                keyword_res = self.keyword_search.search(query, top_k=top_k * 2)
                raw_results = reciprocal_rank_fusion([semantic_res, keyword_res], k=60)
            elif mode == 'semantic':
                raw_results = self.semantic_search.search(expanded_query, top_k=top_k * 2)
            elif mode == 'keyword':
                raw_results = self.keyword_search.search(query, top_k=top_k * 2)
            elif mode == 'face':
                logger.warning("Mode 'face' is not directly supported for text query without identity matching.")
                return []
            else:
                logger.error(f"Unknown search mode: {mode}")
                return []

            # Apply institutional category / project / label_type filtering if requested
            db = self.keyword_search.db
            if category or project or label_type:
                allowed_ids = set(db.get_images_by_taxonomy_filter(
                    category=category, project=project, label_type=label_type
                ))
                raw_results = [(img_id, score) for img_id, score in raw_results if img_id in allowed_ids]

            return self._enrich_results(raw_results[:top_k])
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def search_similar(
        self,
        image_path: str,
        top_k: int = 50,
        category: Optional[str] = None,
        project: Optional[str] = None
    ) -> List[SearchResult]:
        """Find visually similar images with optional taxonomy filters."""
        try:
            raw_results = self.semantic_search.search_by_image(image_path, top_k=top_k * 2)
            db = self.keyword_search.db
            if category or project:
                allowed_ids = set(db.get_images_by_taxonomy_filter(category=category, project=project))
                raw_results = [(img_id, score) for img_id, score in raw_results if img_id in allowed_ids]

            return self._enrich_results(raw_results[:top_k])
        except Exception as e:
            logger.error(f"Similar image search failed for '{image_path}': {e}")
            return []

    def search_face(
        self,
        photo_path: str,
        top_k: int = 50,
        category: Optional[str] = None,
        project: Optional[str] = None
    ) -> List[SearchResult]:
        """Find images containing the face in the photo with optional filters."""
        try:
            face_res = self.face_search.search_by_photo(photo_path, top_k=top_k * 2)
            raw_results = [(img_id, score) for img_id, score, _ in face_res]

            best_scores: Dict[int, float] = {}
            for img_id, score in raw_results:
                if img_id not in best_scores or score > best_scores[img_id]:
                    best_scores[img_id] = score

            db = self.keyword_search.db
            if category or project:
                allowed_ids = set(db.get_images_by_taxonomy_filter(category=category, project=project))
                best_scores = {img_id: sc for img_id, sc in best_scores.items() if img_id in allowed_ids}

            dedup_results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return self._enrich_results(dedup_results)
        except Exception as e:
            logger.error(f"Face search failed for '{photo_path}': {e}")
            return []

    def _enrich_results(self, raw_results: List[Tuple[int, float]]) -> List[SearchResult]:
        """
        Enrich raw (image_id, score) pairs with full metadata from the database.
        """
        if not raw_results:
            return []

        enriched: List[SearchResult] = []
        db = self.keyword_search.db

        for image_id, score in raw_results:
            try:
                img = db.get_image_by_id(image_id)
                if not img:
                    continue

                file_path = img.get('file_path', '')
                basename = os.path.splitext(os.path.basename(file_path))[0]
                thumb_path = f"/thumbnails/{basename}.jpg"

                caption_text = None
                with db.get_connection() as conn:
                    cap_row = conn.execute(
                        "SELECT caption FROM caption WHERE image_id = ?", (image_id,)
                    ).fetchone()
                    if cap_row:
                        caption_text = cap_row['caption']

                tag_rows = db.get_image_tags(image_id)
                tags = [t['tag'] for t in tag_rows]

                face_rows = db.get_image_faces(image_id)
                faces = [
                    {'face_id': f['id'], 'bbox': f['bbox'], 'cluster_id': f.get('cluster_id')}
                    for f in face_rows
                ]

                # Institutional tags
                inst_tags = db.get_image_institutional_tags(image_id)

                enriched.append(SearchResult(
                    image_id=image_id,
                    file_path=file_path,
                    thumbnail_path=thumb_path,
                    score=score,
                    matched_tags=tags,
                    caption=caption_text,
                    faces=faces,
                    institutional_tags=inst_tags,
                ))
            except Exception as e:
                logger.warning(f"Failed to enrich image_id={image_id}: {e}")

        return enriched
