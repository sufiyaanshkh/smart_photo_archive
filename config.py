import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class PathsConfig:
    images_dir: str = "data/images"
    thumbnails_dir: str = "data/thumbnails"
    database_path: str = "data/archive.db"
    faiss_clip_index_path: str = "data/faiss_clip.index"
    faiss_face_index_path: str = "data/faiss_face.index"
    models_cache_dir: str = "models_cache"
    taxonomy_excel_path: str = "data/institutional_taxonomy.xlsx"
    keyword_embeddings_path: str = "data/keyword_embeddings.npy"

@dataclass
class ModelsConfig:
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    ram_model_name: str = "ram_plus_swin_large_14m"
    florence_model_name: str = "microsoft/Florence-2-large"
    insightface_model_name: str = "buffalo_l"

@dataclass
class ProcessingConfig:
    batch_size: int = 64
    det_size: int = 640
    thumbnail_size: int = 256
    num_workers: int = 4

@dataclass
class FaceConfig:
    min_cluster_size: int = 3
    cosine_threshold: float = 0.45
    min_face_size: int = 30

@dataclass
class ClipConfig:
    embedding_dim: int = 768

@dataclass
class FaissConfig:
    nlist: int = 1000
    nprobe: int = 20

@dataclass
class SearchConfig:
    top_k: int = 50
    rrf_k: int = 60

@dataclass
class KeywordMatchingConfig:
    top_n: int = 10
    min_similarity: float = 0.2

@dataclass
class TaxonomyConfig:
    main_sheet: str = "AI_Tagging_Taxonomy"
    yolo_sheet: str = "YOLO_Class_Index"
    object_similarity_threshold: float = 0.22
    context_similarity_threshold: float = 0.25
    scene_similarity_threshold: float = 0.23

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    keyword_matching: KeywordMatchingConfig = field(default_factory=KeywordMatchingConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Loads configuration from a YAML file. Provides defaults if missing.
    """
    if not os.path.exists(config_path):
        return Config()
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = Config()
    
    if "paths" in data:
        config.paths = PathsConfig(**data["paths"])
    if "models" in data:
        config.models = ModelsConfig(**data["models"])
    if "processing" in data:
        config.processing = ProcessingConfig(**data["processing"])
    if "face" in data:
        config.face = FaceConfig(**data["face"])
    if "clip" in data:
        config.clip = ClipConfig(**data["clip"])
    if "faiss" in data:
        config.faiss = FaissConfig(**data["faiss"])
    if "search" in data:
        config.search = SearchConfig(**data["search"])
    if "keyword_matching" in data:
        config.keyword_matching = KeywordMatchingConfig(**data["keyword_matching"])
    if "taxonomy" in data:
        config.taxonomy = TaxonomyConfig(**data["taxonomy"])
        
    return config
