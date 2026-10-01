"""Keyword search using SQLite FTS5 and tags/captions."""

import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class KeywordSearch:
    """Keyword search engine using SQLite FTS5 and tag matching."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Search across tags and captions.
        """
        results: List[Tuple[int, float]] = []
        try:
            # Clean query for FTS5 matching
            clean_q = "".join([c for c in query if c.isalnum() or c.isspace()]).strip()
            
            with self.db.get_connection() as conn:
                # 1. Try FTS5 matching on captions & tags if available
                fts_ids = []
                try:
                    if clean_q:
                        fts_cursor = conn.execute("""
                            SELECT image_id FROM caption
                            WHERE image_id IN (
                                SELECT rowid FROM fts_captions WHERE fts_captions MATCH ?
                            )
                            UNION
                            SELECT image_id FROM tags
                            WHERE id IN (
                                SELECT rowid FROM fts_tags WHERE fts_tags MATCH ?
                            )
                            LIMIT ?
                        """, (clean_q, clean_q, top_k))
                        fts_ids = [r['image_id'] for r in fts_cursor.fetchall()]
                except Exception as fts_err:
                    logger.debug(f"FTS5 match fallback: {fts_err}")

                # 2. Also match via direct LIKE search on tags table for partial word matches
                like_pattern = f"%{clean_q}%" if clean_q else f"%{query}%"
                cursor = conn.execute("""
                    SELECT image_id, MAX(confidence) as max_conf
                    FROM tags
                    WHERE tag LIKE ?
                    GROUP BY image_id
                    ORDER BY max_conf DESC
                    LIMIT ?
                """, (like_pattern, top_k))
                like_rows = cursor.fetchall()

            seen = set()
            for img_id in fts_ids:
                if img_id not in seen:
                    results.append((int(img_id), 1.0))
                    seen.add(img_id)

            for r in like_rows:
                iid = int(r['image_id'])
                if iid not in seen:
                    results.append((iid, float(r['max_conf'] or 0.8)))
                    seen.add(iid)

            return results[:top_k]
        except Exception as e:
            logger.error(f"Keyword search failed for query '{query}': {e}")
            return []
