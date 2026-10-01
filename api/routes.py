import logging
import time
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any

from .schemas import (
    SearchResponse, SearchResult, ImageDetail, ClusterInfo, 
    IdentityCreate, IdentityUpdate, ArchiveStats, TagInfo, FaceInfo,
    InstitutionalTagInfo, TaxonomyKeywordInfo
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search_images(
    request: Request,
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode: hybrid, semantic, keyword"),
    top_k: int = Query(50, description="Number of top results to return"),
    category: Optional[str] = Query(None, description="Filter by IIHS primary category"),
    project: Optional[str] = Query(None, description="Filter by project/event"),
    label_type: Optional[str] = Query(None, description="Filter by label type (object, context, scene)")
):
    """Search for images using text query with optional institutional filters."""
    start_time = time.time()
    logger.info(f"Searching with query='{q}', mode='{mode}', top_k={top_k}, category='{category}', project='{project}'")

    engine = getattr(request.app.state, "search_engine", None)
    results = []
    if engine is not None:
        raw_results = engine.search(
            query=q,
            top_k=top_k,
            mode=mode,
            category=category,
            project=project,
            label_type=label_type,
        )
        for r in raw_results:
            inst_list = [
                InstitutionalTagInfo(
                    keyword_id=it.get('keyword_id', 0),
                    keyword=it.get('normalized_keyword', ''),
                    category=it.get('primary_category'),
                    project=it.get('project'),
                    label_type=it.get('label_type'),
                    similarity_score=it.get('similarity_score'),
                    match_method=it.get('match_method'),
                )
                for it in (r.institutional_tags or [])
            ]
            results.append(SearchResult(
                image_id=r.image_id,
                file_path=r.file_path,
                thumbnail_url=r.thumbnail_path,
                score=r.score,
                tags=r.matched_tags or [],
                caption=r.caption,
                faces=[
                    FaceInfo(
                        face_id=f.get('face_id', 0),
                        bbox=f.get('bbox', [0, 0, 0, 0]) if isinstance(f.get('bbox'), list) else [0, 0, 0, 0],
                        cluster_id=f.get('cluster_id')
                    ) for f in (r.faces or [])
                ],
                institutional_tags=inst_list
            ))
    
    elapsed_ms = (time.time() - start_time) * 1000
    return SearchResponse(
        results=results,
        total_count=len(results),
        query=q,
        mode=mode,
        category=category,
        project=project,
        search_time_ms=elapsed_ms
    )


@router.post("/search/face", response_model=SearchResponse)
async def search_by_face(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(50, description="Top results"),
    category: Optional[str] = Query(None),
    project: Optional[str] = Query(None)
):
    """Search for images containing a specific face uploaded by the user."""
    start_time = time.time()
    logger.info(f"Face search uploaded file: {file.filename}")

    engine = getattr(request.app.state, "search_engine", None)
    results = []

    if engine is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "img.jpg")[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            raw_results = engine.search_face(tmp_path, top_k=top_k, category=category, project=project)
            for r in raw_results:
                results.append(SearchResult(
                    image_id=r.image_id,
                    file_path=r.file_path,
                    thumbnail_url=r.thumbnail_path,
                    score=r.score,
                    tags=r.matched_tags or [],
                    caption=r.caption,
                    faces=[],
                    institutional_tags=[]
                ))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elapsed_ms = (time.time() - start_time) * 1000
    return SearchResponse(
        results=results,
        total_count=len(results),
        query="face_upload",
        mode="face",
        category=category,
        project=project,
        search_time_ms=elapsed_ms
    )


@router.post("/search/similar", response_model=SearchResponse)
async def search_similar_images(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(50, description="Top results"),
    category: Optional[str] = Query(None),
    project: Optional[str] = Query(None)
):
    """Search for images visually similar to an uploaded image."""
    start_time = time.time()
    logger.info(f"Similar image search uploaded file: {file.filename}")

    engine = getattr(request.app.state, "search_engine", None)
    results = []

    if engine is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "img.jpg")[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            raw_results = engine.search_similar(tmp_path, top_k=top_k, category=category, project=project)
            for r in raw_results:
                results.append(SearchResult(
                    image_id=r.image_id,
                    file_path=r.file_path,
                    thumbnail_url=r.thumbnail_path,
                    score=r.score,
                    tags=r.matched_tags or [],
                    caption=r.caption,
                    faces=[],
                    institutional_tags=[]
                ))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elapsed_ms = (time.time() - start_time) * 1000
    return SearchResponse(
        results=results,
        total_count=len(results),
        query="image_upload",
        mode="similar",
        category=category,
        project=project,
        search_time_ms=elapsed_ms
    )


@router.get("/image/{image_id}", response_model=ImageDetail)
async def get_image_detail(request: Request, image_id: int):
    """Get full details of a specific image including institutional tags."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    img = db.get_image_by_id(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = img.get('file_path', '')
    basename = os.path.splitext(os.path.basename(file_path))[0]
    thumb_url = f"/thumbnails/{basename}.jpg"

    caption_text = None
    with db.get_connection() as conn:
        cap_row = conn.execute("SELECT caption FROM caption WHERE image_id = ?", (image_id,)).fetchone()
        if cap_row:
            caption_text = cap_row['caption']

    tags_rows = db.get_image_tags(image_id)
    tags = [TagInfo(tag=t['tag'], source=t.get('source', ''), confidence=t.get('confidence', 1.0)) for t in tags_rows]

    face_rows = db.get_image_faces(image_id)
    faces = [FaceInfo(face_id=f['id'], bbox=[0,0,0,0], cluster_id=f.get('cluster_id')) for f in face_rows]

    inst_rows = db.get_image_institutional_tags(image_id)
    inst_tags = [
        InstitutionalTagInfo(
            keyword_id=r['keyword_id'],
            keyword=r['normalized_keyword'],
            category=r.get('primary_category'),
            project=r.get('project'),
            label_type=r.get('label_type'),
            similarity_score=r.get('similarity_score'),
            match_method=r.get('match_method')
        ) for r in inst_rows
    ]

    return ImageDetail(
        image_id=image_id,
        file_path=file_path,
        thumbnail_url=thumb_url,
        width=img.get('width'),
        height=img.get('height'),
        taken_at=img.get('created_at'),
        caption=caption_text,
        tags=tags,
        faces=faces,
        institutional_tags=inst_tags
    )


@router.get("/image/{image_id}/thumbnail")
async def get_image_thumbnail(request: Request, image_id: int):
    """Serve the thumbnail file for a given image."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        img = db.get_image_by_id(image_id)
        if img:
            base = os.path.splitext(os.path.basename(img['file_path']))[0]
            thumb_path = os.path.join(os.getcwd(), 'data', 'thumbnails', f"{base}.jpg")
            if os.path.exists(thumb_path):
                return FileResponse(thumb_path)

    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("/taxonomy/categories", response_model=List[str])
async def list_taxonomy_categories(request: Request):
    """List all available institutional taxonomy primary categories."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        return db.get_taxonomy_categories()
    return []


@router.get("/taxonomy/projects", response_model=List[str])
async def list_taxonomy_projects(request: Request):
    """List all institutional projects/events in the taxonomy."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        return db.get_taxonomy_projects()
    return []


@router.get("/taxonomy/keywords", response_model=List[TaxonomyKeywordInfo])
async def list_taxonomy_keywords(
    request: Request,
    category: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    label_type: Optional[str] = Query(None)
):
    """List taxonomy keywords with optional filtering."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return []

    keywords = db.get_all_institutional_keywords()
    filtered = []
    for k in keywords:
        if category and k.get('primary_category') != category:
            continue
        if project and k.get('project') != project:
            continue
        if label_type and k.get('label_type') != label_type:
            continue

        filtered.append(TaxonomyKeywordInfo(
            id=k['id'],
            normalized_keyword=k['normalized_keyword'],
            original_keyword=k['original_keyword'],
            primary_category=k.get('primary_category', ''),
            project=k.get('project', ''),
            label_type=k.get('label_type', 'context'),
            candidate_visual_class=k.get('candidate_visual_class'),
            priority=k.get('priority', 'High')
        ))
    return filtered


@router.get("/faces/clusters", response_model=List[ClusterInfo])
async def list_face_clusters(request: Request):
    """List all face clusters."""
    db = getattr(request.app.state, "db", None)
    clusters = []
    if db is not None:
        with db.get_connection() as conn:
            rows = conn.execute('''
                SELECT cluster_id, COUNT(*) as face_count
                FROM faces
                WHERE cluster_id >= 0
                GROUP BY cluster_id
                ORDER BY face_count DESC
            ''').fetchall()
            for r in rows:
                clusters.append(ClusterInfo(
                    cluster_id=r['cluster_id'],
                    face_count=r['face_count'],
                    representative_thumbnail_url=f"/thumbnails/cluster_{r['cluster_id']}.jpg",
                    identity_name=None
                ))
    return clusters


@router.get("/faces/identities")
async def list_identities(request: Request):
    """List all known identities."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        with db.get_connection() as conn:
            rows = conn.execute("SELECT id, name, label FROM identities").fetchall()
            return [dict(r) for r in rows]
    return []


@router.put("/faces/identities/{identity_id}")
async def update_identity(request: Request, identity_id: int, payload: IdentityUpdate):
    """Update identity name or label."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        with db.get_connection() as conn:
            if payload.name:
                conn.execute("UPDATE identities SET name=? WHERE id=?", (payload.name, identity_id))
            if payload.label:
                conn.execute("UPDATE identities SET label=? WHERE id=?", (payload.label, identity_id))
        return {"status": "success", "identity_id": identity_id, "updated": True}
    return {"status": "error", "message": "Database unavailable"}


@router.post("/faces/identities")
async def create_identity(request: Request, payload: IdentityCreate):
    """Create a new identity and link faces to it."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        new_id = db.insert_identity(payload.name, payload.label)
        for fid in payload.face_ids:
            db.link_face_to_identity(fid, new_id)
        return {"status": "success", "identity_id": new_id, "name": payload.name}
    return {"status": "error", "message": "Database unavailable"}


@router.get("/stats", response_model=ArchiveStats)
async def get_stats(request: Request):
    """Get archive statistics including institutional taxonomy counts."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        with db.get_connection() as conn:
            total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            total_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            total_clusters = conn.execute("SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id >= 0").fetchone()[0]
            total_identities = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
            total_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
            unique_tags = conn.execute("SELECT COUNT(DISTINCT tag) FROM tags").fetchone()[0]
            total_ik = conn.execute("SELECT COUNT(*) FROM institutional_keywords").fetchone()[0]
            total_cats = conn.execute("SELECT COUNT(DISTINCT primary_category) FROM institutional_keywords WHERE primary_category != ''").fetchone()[0]
            total_projs = conn.execute("SELECT COUNT(DISTINCT project) FROM institutional_keywords WHERE project != ''").fetchone()[0]

        return ArchiveStats(
            total_images=total_images,
            total_faces=total_faces,
            total_clusters=total_clusters,
            total_identities=total_identities,
            total_tags=total_tags,
            unique_tags=unique_tags,
            total_institutional_keywords=total_ik,
            total_categories=total_cats,
            total_projects=total_projs
        )

    return ArchiveStats(
        total_images=0, total_faces=0, total_clusters=0,
        total_identities=0, total_tags=0, unique_tags=0
    )
