"""
Pipeline Orchestrator — runs all processing stages with fault-tolerant checkpointing.

Each stage:
  1. Queries the DB for already-processed image IDs.
  2. Processes only remaining images in batches.
  3. Persists results to SQLite after each batch.
  4. Marks the batch as checkpointed so restarts skip completed work.
"""

import logging
import os
import glob
import signal
import sys
import json
import struct
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    from config import Config, load_config
    from storage.database import DatabaseManager
    from storage.vector_store import CLIPVectorStore, FaceVectorStore
    from storage.thumbnail import generate_thumbnail
    from pipeline.exif_extractor import extract_exif
    from pipeline.clip_embedder import CLIPEmbedder
    from pipeline.tagger import ImageTagger
    from pipeline.captioner import ImageCaptioner
    from pipeline.face_processor import FaceProcessor
    from pipeline.face_clusterer import FaceClusterer
    from pipeline.keyword_matcher import KeywordMatcher
except (ImportError, ValueError):
    from ..config import Config, load_config
    from ..storage.database import DatabaseManager
    from ..storage.vector_store import CLIPVectorStore, FaceVectorStore
    from ..storage.thumbnail import generate_thumbnail
    from .exif_extractor import extract_exif
    from .clip_embedder import CLIPEmbedder
    from .tagger import ImageTagger
    from .captioner import ImageCaptioner
    from .face_processor import FaceProcessor
    from .face_clusterer import FaceClusterer
    from .keyword_matcher import KeywordMatcher

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a numpy float32 array to bytes."""
    return embedding.astype(np.float32).tobytes()


def _bytes_to_embedding(data: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes to a numpy float32 array."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)


class PipelineOrchestrator:
    """
    Main pipeline orchestrator for the Smart Photo Archive System.

    Runs all ML stages in order with per-image checkpointing so that
    the pipeline can be interrupted and resumed without reprocessing.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        self.config: Config = load_config(config_path)
        self.shutdown_requested: bool = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Database
        self.db = DatabaseManager(self.config.paths.database_path)
        self.db.init_db()

        # ML modules — lazily initialised in initialize_modules()
        self.clip_embedder: Optional[CLIPEmbedder] = None
        self.tagger: Optional[ImageTagger] = None
        self.captioner: Optional[ImageCaptioner] = None
        self.face_processor: Optional[FaceProcessor] = None
        self.face_clusterer: Optional[FaceClusterer] = None
        self.keyword_matcher: Optional[KeywordMatcher] = None

        logger.info("PipelineOrchestrator initialised (config=%s).", config_path)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.warning("Shutdown signal received — finishing current batch then exiting.")
        self.shutdown_requested = True

    def initialize_modules(self, stages: Optional[List[str]] = None):
        """
        Load ML models onto GPU/CPU.

        If *stages* is provided only the models required for those stages
        are loaded, saving memory.
        """
        all_stages = stages is None

        if all_stages or 'clip' in stages or 'keyword_matching' in stages:
            logger.info("Loading CLIP embedder…")
            self.clip_embedder = CLIPEmbedder(
                model_name=self.config.models.clip_model_name,
                pretrained=self.config.models.clip_pretrained,
            )

        if all_stages or 'tags' in stages:
            logger.info("Loading RAM++ tagger…")
            self.tagger = ImageTagger()

        if all_stages or 'captions' in stages:
            logger.info("Loading Florence-2 captioner…")
            self.captioner = ImageCaptioner(
                model_name=self.config.models.florence_model_name,
            )

        if all_stages or 'faces' in stages or 'face_clustering' in stages:
            logger.info("Loading InsightFace processor…")
            self.face_processor = FaceProcessor(
                model_name=self.config.models.insightface_model_name,
                det_size=(self.config.processing.det_size, self.config.processing.det_size),
            )
            self.face_clusterer = FaceClusterer(
                min_cluster_size=self.config.face.min_cluster_size,
            )

        if all_stages or 'keyword_matching' in stages:
            if self.clip_embedder is not None:
                self.keyword_matcher = KeywordMatcher(self.clip_embedder)

        logger.info("Model loading complete.")

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------

    def discover_images(self, images_dir: Optional[str] = None) -> List[str]:
        """Recursively discover supported image files."""
        root = images_dir or self.config.paths.images_dir
        image_paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(dirpath, fname))
        image_paths.sort()
        logger.info("Discovered %d images in %s", len(image_paths), root)
        return image_paths

    # ------------------------------------------------------------------
    # Stage: register images + EXIF
    # ------------------------------------------------------------------

    def _run_exif_stage(self, image_paths: List[str]):
        """Extract EXIF metadata and register images in the database."""
        logger.info("=== Stage: EXIF Extraction ===")
        processed = set(self.db.get_processed_ids('exif'))
        # We need image_id → path mapping; register unprocessed images first.
        remaining = []
        for p in image_paths:
            img = self.db.insert_image(p, {})  # register path, get ID
            if img not in processed:
                remaining.append((img, p))

        logger.info("%d already checkpointed, %d remaining.", len(processed), len(remaining))

        for image_id, path in tqdm(remaining, desc="EXIF"):
            if self.shutdown_requested:
                logger.warning("Shutdown requested — halting EXIF stage.")
                return
            try:
                meta = extract_exif(path)
                # Update the image row with EXIF data
                with self.db.get_connection() as conn:
                    conn.execute(
                        """UPDATE images SET width=?, height=?
                           WHERE id=?""",
                        (meta.get('width'), meta.get('height'), image_id),
                    )
                self.db.mark_processed('exif', [image_id])
            except Exception as e:
                logger.error("EXIF failed for %s: %s", path, e)

    # ------------------------------------------------------------------
    # Stage: thumbnails
    # ------------------------------------------------------------------

    def _run_thumbnail_stage(self, image_paths: List[str]):
        """Generate thumbnails for all images."""
        logger.info("=== Stage: Thumbnail Generation ===")
        os.makedirs(self.config.paths.thumbnails_dir, exist_ok=True)
        processed = set(self.db.get_processed_ids('thumbnails'))

        remaining = []
        for p in image_paths:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT id FROM images WHERE file_path=?", (p,)).fetchone()
            if row and row['id'] not in processed:
                remaining.append((row['id'], p))

        logger.info("%d thumbnails remaining.", len(remaining))
        for image_id, path in tqdm(remaining, desc="Thumbnails"):
            if self.shutdown_requested:
                return
            try:
                generate_thumbnail(
                    path,
                    self.config.paths.thumbnails_dir,
                    size=self.config.processing.thumbnail_size,
                )
                self.db.mark_processed('thumbnails', [image_id])
            except Exception as e:
                logger.error("Thumbnail failed for %s: %s", path, e)

    # ------------------------------------------------------------------
    # Stage: CLIP embeddings
    # ------------------------------------------------------------------

    def _run_clip_stage(self, image_paths: List[str]):
        """Compute CLIP image embeddings and store in DB."""
        logger.info("=== Stage: CLIP Embeddings ===")
        assert self.clip_embedder is not None, "CLIPEmbedder not loaded."

        processed = set(self.db.get_processed_ids('clip'))
        remaining = []
        for p in image_paths:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT id FROM images WHERE file_path=?", (p,)).fetchone()
            if row and row['id'] not in processed:
                remaining.append((row['id'], p))

        logger.info("%d CLIP embeddings remaining.", len(remaining))
        bs = self.config.processing.batch_size
        for i in tqdm(range(0, len(remaining), bs), desc="CLIP"):
            if self.shutdown_requested:
                return
            batch = remaining[i:i + bs]
            paths = [p for _, p in batch]
            ids = [iid for iid, _ in batch]
            try:
                embeddings = self.clip_embedder.embed_images_batch(paths, batch_size=bs)
                for j, image_id in enumerate(ids):
                    self.db.insert_clip_embedding(image_id, _embedding_to_bytes(embeddings[j]))
                self.db.mark_processed('clip', ids)
            except Exception as e:
                logger.error("CLIP batch failed: %s", e)

    # ------------------------------------------------------------------
    # Stage: tagging (RAM++)
    # ------------------------------------------------------------------

    def _run_tags_stage(self, image_paths: List[str]):
        """Run RAM++ to generate open-vocabulary tags."""
        logger.info("=== Stage: RAM++ Tagging ===")
        assert self.tagger is not None, "ImageTagger not loaded."

        processed = set(self.db.get_processed_ids('tags'))
        remaining = []
        for p in image_paths:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT id FROM images WHERE file_path=?", (p,)).fetchone()
            if row and row['id'] not in processed:
                remaining.append((row['id'], p))

        logger.info("%d images remaining for tagging.", len(remaining))
        for image_id, path in tqdm(remaining, desc="Tags"):
            if self.shutdown_requested:
                return
            try:
                tags = self.tagger.tag_image(path)
                tag_rows = [(tag, 'ram++', conf) for tag, conf in tags]
                self.db.insert_tags(image_id, tag_rows)
                self.db.mark_processed('tags', [image_id])
            except Exception as e:
                logger.error("Tagging failed for %s: %s", path, e)

    # ------------------------------------------------------------------
    # Stage: captioning (Florence-2)
    # ------------------------------------------------------------------

    def _run_captions_stage(self, image_paths: List[str]):
        """Run Florence-2 to generate image captions."""
        logger.info("=== Stage: Florence-2 Captioning ===")
        assert self.captioner is not None, "ImageCaptioner not loaded."

        processed = set(self.db.get_processed_ids('captions'))
        remaining = []
        for p in image_paths:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT id FROM images WHERE file_path=?", (p,)).fetchone()
            if row and row['id'] not in processed:
                remaining.append((row['id'], p))

        logger.info("%d images remaining for captioning.", len(remaining))
        bs = 16
        for i in tqdm(range(0, len(remaining), bs), desc="Captions"):
            if self.shutdown_requested:
                return
            batch = remaining[i:i + bs]
            paths = [p for _, p in batch]
            ids = [iid for iid, _ in batch]
            try:
                captions = self.captioner.caption_images_batch(paths, batch_size=bs)
                for j, image_id in enumerate(ids):
                    self.db.insert_caption(image_id, captions[j])
                self.db.mark_processed('captions', ids)
            except Exception as e:
                logger.error("Captioning batch failed: %s", e)

    # ------------------------------------------------------------------
    # Stage: face detection + embedding
    # ------------------------------------------------------------------

    def _run_faces_stage(self, image_paths: List[str]):
        """Detect faces with InsightFace and store embeddings."""
        logger.info("=== Stage: InsightFace Detection ===")
        assert self.face_processor is not None, "FaceProcessor not loaded."

        processed = set(self.db.get_processed_ids('faces'))
        remaining = []
        for p in image_paths:
            with self.db.get_connection() as conn:
                row = conn.execute("SELECT id FROM images WHERE file_path=?", (p,)).fetchone()
            if row and row['id'] not in processed:
                remaining.append((row['id'], p))

        logger.info("%d images remaining for face detection.", len(remaining))
        for image_id, path in tqdm(remaining, desc="Faces"):
            if self.shutdown_requested:
                return
            try:
                faces = self.face_processor.process_image(path)
                for face in faces:
                    bbox_json = json.dumps([float(x) for x in face.bbox])
                    self.db.insert_face(
                        image_id=image_id,
                        bbox=bbox_json,
                        embedding_bytes=_embedding_to_bytes(face.embedding),
                        confidence=face.det_score,
                    )
                self.db.mark_processed('faces', [image_id])
            except Exception as e:
                logger.error("Face detection failed for %s: %s", path, e)

    # ------------------------------------------------------------------
    # Post-processing: face clustering
    # ------------------------------------------------------------------

    def _run_face_clustering(self):
        """Cluster all face embeddings with HDBSCAN."""
        logger.info("=== Post-Processing: Face Clustering ===")
        assert self.face_clusterer is not None, "FaceClusterer not loaded."

        face_rows = self.db.get_all_face_embeddings()
        if not face_rows:
            logger.warning("No face embeddings found — skipping clustering.")
            return

        face_ids = [r['id'] for r in face_rows]
        embeddings = np.array(
            [_bytes_to_embedding(r['embedding'], 512) for r in face_rows],
            dtype=np.float32,
        )

        logger.info("Clustering %d face embeddings…", len(embeddings))
        labels = self.face_clusterer.cluster(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        logger.info("Found %d clusters, %d noise faces.", n_clusters, n_noise)

        for face_id, cluster_id in zip(face_ids, labels):
            self.db.update_face_cluster(face_id, int(cluster_id))

    # ------------------------------------------------------------------
    # Post-processing: keyword matching
    # ------------------------------------------------------------------

    def _run_keyword_matching(self, keywords_npy_path: Optional[str] = None):
        """Match institutional keywords to images using CLIP zero-shot and taxonomy rules."""
        logger.info("=== Post-Processing: Institutional Taxonomy Matching ===")
        if self.keyword_matcher is None:
            logger.warning("KeywordMatcher not initialised — skipping.")
            return

        npy_path = keywords_npy_path or getattr(self.config.paths, 'keyword_embeddings_path', 'data/keyword_embeddings.npy')
        loaded = self.keyword_matcher.load_precomputed_embeddings(npy_path)
        if not loaded:
            logger.warning("No precomputed keyword embeddings found at %s — skipping.", npy_path)
            return

        # Iterate over all images with CLIP embeddings
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT image_id, embedding FROM clip_embeddings").fetchall()

        logger.info(
            "Matching %d institutional keywords against %d images…",
            len(self.keyword_matcher.keyword_list), len(rows)
        )

        tax_thresholds = {
            'object': getattr(self.config.taxonomy, 'object_similarity_threshold', 0.22),
            'context': getattr(self.config.taxonomy, 'context_similarity_threshold', 0.25),
            'scene': getattr(self.config.taxonomy, 'scene_similarity_threshold', 0.23),
            'metadata': 0.30,
            'entity_metadata': 0.28,
        }

        for row in tqdm(rows, desc="Taxonomy Matching"):
            image_id = row['image_id']
            emb = _bytes_to_embedding(row['embedding'], 768)
            matches = self.keyword_matcher.match_image_taxonomy(
                emb,
                top_n=self.config.keyword_matching.top_n,
                default_min_similarity=self.config.keyword_matching.min_similarity,
                type_thresholds=tax_thresholds,
            )
            if matches:
                # 1. Insert into dedicated image_institutional_tags table
                inst_tags = [
                    (image_id, m['keyword_id'], float(m['score']), m['match_method'])
                    for m in matches
                ]
                self.db.insert_image_institutional_tags(inst_tags)

                # 2. Also insert into general tags table for FTS5 full-text indexing
                tag_rows = [
                    (m['keyword'], f"iihs_{m['label_type']}", float(m['score']))
                    for m in matches
                ]
                self.db.insert_tags(image_id, tag_rows)

    # ------------------------------------------------------------------
    # Post-processing: build FAISS indices
    # ------------------------------------------------------------------

    def _build_faiss_indices(self):
        """Build and save FAISS indices for CLIP and face embeddings."""
        logger.info("=== Post-Processing: Building FAISS Indices ===")

        # CLIP index
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT image_id, embedding FROM clip_embeddings").fetchall()

        if rows:
            ids = [r['image_id'] for r in rows]
            embeddings = np.array(
                [_bytes_to_embedding(r['embedding'], 768) for r in rows],
                dtype=np.float32,
            )
            clip_store = CLIPVectorStore()
            clip_store.build_index(
                embeddings, ids,
                use_ivf=(len(rows) >= self.config.faiss.nlist),
                nlist=self.config.faiss.nlist,
            )
            os.makedirs(os.path.dirname(self.config.paths.faiss_clip_index_path) or '.', exist_ok=True)
            clip_store.save_index(self.config.paths.faiss_clip_index_path)
            logger.info("CLIP FAISS index saved (%d vectors).", len(ids))

            # Save ID map alongside the index
            id_map_path = self.config.paths.faiss_clip_index_path + '.idmap.npy'
            np.save(id_map_path, {'id_map': clip_store.id_map, 'reverse_id_map': clip_store.reverse_id_map})
        else:
            logger.warning("No CLIP embeddings — skipping CLIP FAISS index.")

        # Face index
        face_rows = self.db.get_all_face_embeddings()
        if face_rows:
            face_ids = [r['id'] for r in face_rows]
            face_embs = np.array(
                [_bytes_to_embedding(r['embedding'], 512) for r in face_rows],
                dtype=np.float32,
            )
            face_store = FaceVectorStore()
            face_store.build_index(face_embs, face_ids, use_ivf=False)
            face_store.save_index(self.config.paths.faiss_face_index_path)

            id_map_path = self.config.paths.faiss_face_index_path + '.idmap.npy'
            np.save(id_map_path, {'id_map': face_store.id_map, 'reverse_id_map': face_store.reverse_id_map})
            logger.info("Face FAISS index saved (%d vectors).", len(face_ids))
        else:
            logger.warning("No face embeddings — skipping face FAISS index.")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    STAGE_ORDER = [
        'exif', 'thumbnails', 'clip', 'tags', 'captions', 'faces',
        'face_clustering', 'keyword_matching', 'build_faiss_indices',
    ]

    def run_full_pipeline(
        self,
        images_dir: Optional[str] = None,
        stages: Optional[List[str]] = None,
        keywords_npy_path: Optional[str] = None,
    ):
        """
        Run the complete processing pipeline.

        Args:
            images_dir: Override for images directory.
            stages: List of stage names to run, or None for all.
            keywords_npy_path: Path to pre-computed keyword embeddings (.npy).
        """
        image_paths = self.discover_images(images_dir)
        if not image_paths:
            logger.warning("No images found — nothing to do.")
            return

        active_stages = stages or self.STAGE_ORDER
        self.initialize_modules(stages=active_stages)

        stage_runners = {
            'exif': lambda: self._run_exif_stage(image_paths),
            'thumbnails': lambda: self._run_thumbnail_stage(image_paths),
            'clip': lambda: self._run_clip_stage(image_paths),
            'tags': lambda: self._run_tags_stage(image_paths),
            'captions': lambda: self._run_captions_stage(image_paths),
            'faces': lambda: self._run_faces_stage(image_paths),
            'face_clustering': lambda: self._run_face_clustering(),
            'keyword_matching': lambda: self._run_keyword_matching(keywords_npy_path),
            'build_faiss_indices': lambda: self._build_faiss_indices(),
        }

        for stage in active_stages:
            if self.shutdown_requested:
                logger.warning("Pipeline halted by shutdown signal after stage list.")
                break
            runner = stage_runners.get(stage)
            if runner is None:
                logger.warning("Unknown stage '%s' — skipping.", stage)
                continue
            try:
                runner()
            except Exception as e:
                logger.error("Stage '%s' failed: %s", stage, e, exc_info=True)

        # Print summary
        with self.db.get_connection() as conn:
            n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            n_clusters = conn.execute(
                "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id >= 0"
            ).fetchone()[0]
            n_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Images processed : %d", n_images)
        logger.info("  Faces detected   : %d", n_faces)
        logger.info("  Face clusters    : %d", n_clusters)
        logger.info("  Tags generated   : %d", n_tags)
        logger.info("=" * 60)
