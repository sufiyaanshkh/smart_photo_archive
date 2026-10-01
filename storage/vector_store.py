import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss is not installed. Using pure NumPy vector store fallback.")


class VectorStore:
    """Vector store supporting FAISS with pure NumPy fallback."""

    def __init__(self, dimension: int, index_type: str = "Flat", use_gpu: bool = False, metric: Any = None):
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.metric = metric
        self.index = None
        self.id_map: Dict[int, int] = {}          # internal ID -> external ID
        self.reverse_id_map: Dict[int, int] = {}  # external ID -> internal ID
        # NumPy fallback storage
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[int] = []

    def build_index(self, embeddings_matrix: np.ndarray, ids: List[int], use_ivf: bool = True, nlist: int = 1000):
        """Builds vector index from an embeddings matrix."""
        if len(embeddings_matrix) == 0:
            logger.warning("Empty embeddings matrix provided. Not building index.")
            return

        mat = np.ascontiguousarray(embeddings_matrix, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        if HAS_FAISS:
            metric_val = faiss.METRIC_INNER_PRODUCT
            if use_ivf and len(mat) >= nlist:
                quantizer = faiss.IndexFlat(self.dimension, metric_val)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_val)
                self.index.train(mat)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)

            self.index.add(mat)

            start_id = len(self.id_map)
            for i, ext_id in enumerate(ids):
                internal_id = start_id + i
                self.id_map[internal_id] = ext_id
                self.reverse_id_map[ext_id] = internal_id

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Could not transfer index to GPU: {e}. Using CPU.")
        else:
            self._matrix = mat
            self._ids = list(ids)
            for i, ext_id in enumerate(ids):
                self.id_map[i] = ext_id
                self.reverse_id_map[ext_id] = i

    def load_index(self, path: str):
        """Loads an index from disk."""
        npy_fallback = path if path.endswith('.npy') else path + ".npy"
        if not os.path.exists(path) and not os.path.exists(npy_fallback):
            logger.warning(f"Index file {path} not found.")
            return

        if HAS_FAISS and os.path.exists(path):
            if self.use_gpu:
                self.index = faiss.read_index(path)
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Could not transfer index to GPU: {e}.")
            else:
                self.index = faiss.read_index(path)
        else:
            if os.path.exists(npy_fallback):
                data = np.load(npy_fallback, allow_pickle=True).item()
                self._matrix = data.get('matrix')
                self._ids = data.get('ids', [])
                for i, ext_id in enumerate(self._ids):
                    self.id_map[i] = ext_id
                    self.reverse_id_map[ext_id] = i

    def save_index(self, path: str):
        """Saves an index to disk."""
        if HAS_FAISS and self.index is not None:
            idx_to_save = self.index
            if self.use_gpu:
                try:
                    idx_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception as e:
                    logger.error(f"Failed to copy index from GPU: {e}")
                    return
            faiss.write_index(idx_to_save, path)
        elif self._matrix is not None:
            npy_fallback = path + ".npy"
            np.save(npy_fallback, {'matrix': self._matrix, 'ids': self._ids})

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> List[Tuple[int, float]]:
        """Searches the index for the top-k nearest neighbors."""
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)

        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if HAS_FAISS and self.index is not None:
            distances, indices = self.index.search(q, top_k)
            results = []
            for i, internal_id in enumerate(indices[0]):
                if internal_id == -1:
                    continue
                ext_id = self.id_map.get(internal_id)
                if ext_id is not None:
                    results.append((ext_id, float(distances[0][i])))
            return results
        elif self._matrix is not None and len(self._matrix) > 0:
            sims = np.dot(self._matrix, q.T).squeeze()
            if sims.ndim == 0:
                sims = np.array([sims])
            top_k = min(top_k, len(sims))
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                ext_id = self.id_map.get(idx, self._ids[idx] if idx < len(self._ids) else idx)
                results.append((ext_id, float(sims[idx])))
            return results

        return []

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Adds new vectors to the index dynamically."""
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = v / norms

        if HAS_FAISS and self.index is not None:
            start_id = self.index.ntotal
            self.index.add(v)
            for i, ext_id in enumerate(ids):
                internal_id = start_id + i
                self.id_map[internal_id] = ext_id
                self.reverse_id_map[ext_id] = internal_id
        else:
            if self._matrix is None:
                self._matrix = v
                self._ids = list(ids)
            else:
                self._matrix = np.vstack([self._matrix, v])
                self._ids.extend(ids)
            for i, ext_id in enumerate(self._ids):
                self.id_map[i] = ext_id
                self.reverse_id_map[ext_id] = i

    def get_dimension(self) -> int:
        return self.dimension


class CLIPVectorStore(VectorStore):
    """Vector store configured specifically for CLIP 768-d embeddings."""
    def __init__(self, use_gpu: bool = False):
        super().__init__(dimension=768, index_type="IVFFlat", use_gpu=use_gpu)


class FaceVectorStore(VectorStore):
    """Vector store configured specifically for Face 512-d embeddings."""
    def __init__(self, use_gpu: bool = False):
        super().__init__(dimension=512, index_type="Flat", use_gpu=use_gpu)
