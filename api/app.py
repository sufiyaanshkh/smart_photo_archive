"""
FastAPI application for the Smart Photo Archive System.

Initialises all backend components (DB, FAISS indices, CLIP model)
on startup and tears them down on shutdown.
"""

import logging
import os

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from config import load_config
    from storage.database import DatabaseManager
    from storage.vector_store import CLIPVectorStore, FaceVectorStore
    from pipeline.clip_embedder import CLIPEmbedder
    from pipeline.face_processor import FaceProcessor
    from search.semantic import SemanticSearch
    from search.keyword import KeywordSearch
    from search.face_search import FaceSearch
    from search.engine import SearchEngine
    from api.routes import router
except (ImportError, ValueError):
    from ..config import load_config
    from ..storage.database import DatabaseManager
    from ..storage.vector_store import CLIPVectorStore, FaceVectorStore
    from ..pipeline.clip_embedder import CLIPEmbedder
    from ..pipeline.face_processor import FaceProcessor
    from ..search.semantic import SemanticSearch
    from ..search.keyword import KeywordSearch
    from ..search.face_search import FaceSearch
    from ..search.engine import SearchEngine
    from .routes import router

logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("SPA_CONFIG", "config.yaml")


def _load_vector_store(index_path: str, store_cls):
    """Load a FAISS or NumPy vector store and its companion id-map from disk."""
    store = store_cls()
    if os.path.exists(index_path) or os.path.exists(index_path + ".npy"):
        store.load_index(index_path)
        idmap_path = index_path + ".idmap.npy"
        if os.path.exists(idmap_path):
            data = np.load(idmap_path, allow_pickle=True).item()
            store.id_map = data.get("id_map", {})
            store.reverse_id_map = data.get("reverse_id_map", {})
        num_vecs = store.index.ntotal if getattr(store, 'index', None) else (len(store._matrix) if getattr(store, '_matrix', None) is not None else 0)
        logger.info("Loaded vector index from %s (%d vectors).", index_path, num_vecs)
    else:
        logger.warning("Vector index file %s not found — search will return empty results.", index_path)
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — initialise all heavy components on startup,
    release them on shutdown.
    """
    config = load_config(CONFIG_PATH)
    logger.info("Starting Smart Photo Archive API (config=%s)…", CONFIG_PATH)

    # ---- Database ----
    db = DatabaseManager(config.paths.database_path)
    db.init_db()
    app.state.db = db

    # ---- FAISS indices ----
    clip_store = _load_vector_store(config.paths.faiss_clip_index_path, CLIPVectorStore)
    face_store = _load_vector_store(config.paths.faiss_face_index_path, FaceVectorStore)
    app.state.clip_store = clip_store
    app.state.face_store = face_store

    # ---- CLIP model (for encoding text queries at search time) ----
    clip_embedder = CLIPEmbedder(
        model_name=config.models.clip_model_name,
        pretrained=config.models.clip_pretrained,
    )
    app.state.clip_embedder = clip_embedder

    # ---- Face processor (for "search by face" uploads) ----
    face_processor = FaceProcessor(
        model_name=config.models.insightface_model_name,
        det_size=(config.processing.det_size, config.processing.det_size),
    )
    app.state.face_processor = face_processor

    # ---- Search engines ----
    semantic = SemanticSearch(clip_embedder, clip_store)
    keyword = KeywordSearch(db)
    face_search = FaceSearch(face_processor, face_store, db)
    engine = SearchEngine(semantic, keyword, face_search, fusion_method="rrf")

    app.state.search_engine = engine
    app.state.config = config

    logger.info("All components ready.")
    yield

    # Cleanup
    logger.info("Shutting down Smart Photo Archive API…")
    app.state.db = None
    app.state.clip_store = None
    app.state.face_store = None
    app.state.clip_embedder = None
    app.state.search_engine = None


# ---- Create the application ----

app = FastAPI(
    title="Smart Photo Archive API",
    description="Search institutional photo archives by text, visual similarity, and faces.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Static mount for thumbnails (created at pipeline time)
thumbnails_dir = os.environ.get("SPA_THUMBNAILS_DIR", "data/thumbnails")
os.makedirs(thumbnails_dir, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=thumbnails_dir), name="thumbnails")
