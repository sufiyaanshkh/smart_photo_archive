"""Face-based search functionality."""

import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class FaceSearch:
    """Search engine for finding faces in images."""
    
    def __init__(self, face_processor: Any, face_vector_store: Any, db: Any) -> None:
        """
        Initialize face search engine.
        
        Args:
            face_processor: Instance for extracting faces.
            face_vector_store: Vector store for face embeddings.
            db: Database instance.
        """
        self.face_processor = face_processor
        self.face_vector_store = face_vector_store
        self.db = db
        
    def search_by_photo(self, photo_path: str, top_k: int = 50) -> List[Tuple[int, float, List[float]]]:
        """
        Find similar faces using a query photo.
        
        Args:
            photo_path: Path to query photo.
            top_k: Number of results.
            
        Returns:
            List of (image_id, score, face_bbox) tuples.
        """
        try:
            faces = self.face_processor.extract_faces(photo_path)
            if not faces:
                logger.warning(f"No faces found in query image: {photo_path}")
                return []
                
            # Use the first face found in query
            query_embedding = faces[0]['embedding']
            vector_results = self.face_vector_store.search(query_embedding, top_k=top_k)
            
            results = []
            for face_id, score in vector_results:
                sql = "SELECT image_id, bbox_x, bbox_y, bbox_w, bbox_h FROM faces WHERE id = ?"
                rows = self.db.execute(sql, (face_id,))
                for row in rows:
                    image_id = row['image_id'] if isinstance(row, dict) else row[0]
                    if isinstance(row, dict):
                        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
                    else:
                        bbox = [row[1], row[2], row[3], row[4]]
                    results.append((int(image_id), float(score), bbox))
            return results
        except Exception as e:
            logger.error(f"Face search by photo failed for '{photo_path}': {e}")
            return []
            
    def search_by_identity(self, identity_id: int, top_k: int = 50) -> List[Tuple[int, List[float]]]:
        """
        Find images containing a specific identity.
        
        Args:
            identity_id: The identity ID.
            top_k: Max results.
            
        Returns:
            List of (image_id, face_bbox).
        """
        try:
            sql = """
            SELECT image_id, bbox_x, bbox_y, bbox_w, bbox_h 
            FROM faces 
            WHERE identity_id = ? 
            LIMIT ?
            """
            rows = self.db.execute(sql, (identity_id, top_k))
            results = []
            for row in rows:
                if isinstance(row, dict):
                    image_id = row['image_id']
                    bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
                else:
                    image_id = row[0]
                    bbox = [row[1], row[2], row[3], row[4]]
                results.append((int(image_id), bbox))
            return results
        except Exception as e:
            logger.error(f"Face search by identity failed for id {identity_id}: {e}")
            return []
            
    def search_by_cluster(self, cluster_id: int) -> List[Tuple[int, List[float]]]:
        """
        Find images in a face cluster.
        
        Args:
            cluster_id: The cluster ID.
            
        Returns:
            List of (image_id, face_bbox).
        """
        try:
            sql = """
            SELECT image_id, bbox_x, bbox_y, bbox_w, bbox_h 
            FROM faces 
            WHERE cluster_id = ?
            """
            rows = self.db.execute(sql, (cluster_id,))
            results = []
            for row in rows:
                if isinstance(row, dict):
                    image_id = row['image_id']
                    bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
                else:
                    image_id = row[0]
                    bbox = [row[1], row[2], row[3], row[4]]
                results.append((int(image_id), bbox))
            return results
        except Exception as e:
            logger.error(f"Face search by cluster failed for id {cluster_id}: {e}")
            return []
