from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TagInfo(BaseModel):
    tag: str = Field(..., description="The string tag.")
    source: str = Field(..., description="Source of the tag (e.g., 'ram++', 'iihs_object').")
    confidence: float = Field(..., description="Confidence score of the tag.")

class InstitutionalTagInfo(BaseModel):
    keyword_id: int
    keyword: str
    category: Optional[str] = None
    project: Optional[str] = None
    label_type: Optional[str] = None
    similarity_score: Optional[float] = None
    match_method: Optional[str] = None

class FaceInfo(BaseModel):
    face_id: int = Field(..., description="Unique ID for the face.")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2].")
    identity_name: Optional[str] = Field(None, description="Name of the recognized identity.")
    cluster_id: Optional[int] = Field(None, description="ID of the cluster this face belongs to.")
    confidence: Optional[float] = Field(None, description="Confidence of the face recognition.")

class SearchResult(BaseModel):
    image_id: int = Field(..., description="Unique ID of the image.")
    file_path: str = Field(..., description="Path to the original image file.")
    thumbnail_url: str = Field(..., description="URL to the generated thumbnail.")
    score: float = Field(..., description="Relevance score.")
    tags: List[str] = Field(default_factory=list, description="List of associated tags.")
    caption: Optional[str] = Field(None, description="Image caption.")
    faces: Optional[List[FaceInfo]] = Field(default_factory=list, description="List of faces found in the image.")
    institutional_tags: Optional[List[InstitutionalTagInfo]] = Field(default_factory=list, description="Matched institutional taxonomy tags.")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="List of search results.")
    total_count: int = Field(..., description="Total number of results found.")
    query: Optional[str] = Field(None, description="The query used for search.")
    mode: str = Field(..., description="Search mode used (e.g., hybrid, semantic).")
    category: Optional[str] = Field(None, description="Category filter applied.")
    project: Optional[str] = Field(None, description="Project filter applied.")
    search_time_ms: float = Field(..., description="Time taken to perform search in milliseconds.")

class ImageDetail(BaseModel):
    image_id: int
    file_path: str
    thumbnail_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    taken_at: Optional[str] = None
    camera_model: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    caption: Optional[str] = None
    tags: List[TagInfo] = Field(default_factory=list)
    faces: List[FaceInfo] = Field(default_factory=list)
    institutional_tags: List[InstitutionalTagInfo] = Field(default_factory=list)

class ClusterInfo(BaseModel):
    cluster_id: int
    face_count: int
    representative_thumbnail_url: str
    identity_name: Optional[str]

class ArchiveStats(BaseModel):
    total_images: int
    total_faces: int
    total_clusters: int
    total_identities: int
    total_tags: int
    unique_tags: int
    total_institutional_keywords: int = 0
    total_categories: int = 0
    total_projects: int = 0

class IdentityCreate(BaseModel):
    name: str
    label: str
    face_ids: List[int]

class IdentityUpdate(BaseModel):
    name: Optional[str] = None
    label: Optional[str] = None

class TaxonomyKeywordInfo(BaseModel):
    id: int
    normalized_keyword: str
    original_keyword: str
    primary_category: str
    project: str
    label_type: str
    candidate_visual_class: Optional[str] = None
    priority: str
