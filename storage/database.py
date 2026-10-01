import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    SQLite database manager for Smart Photo Archive System.
    Provides schema initialization and methods for interacting with image metadata, embeddings, and tags.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    @contextmanager
    def get_connection(self):
        """Context manager for SQLite connections with thread safety enabled."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
            
    def init_db(self):
        """Initializes the database schema."""
        schema = '''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER,
            height INTEGER,
            format TEXT,
            mode TEXT,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            tag TEXT NOT NULL,
            source TEXT,
            confidence REAL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        
        CREATE TABLE IF NOT EXISTS caption (
            image_id INTEGER PRIMARY KEY,
            caption TEXT,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_captions USING fts5(
            caption, content=caption, content_rowid=image_id
        );
        
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_tags USING fts5(
            tag, content=tags, content_rowid=id
        );
        
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            bbox TEXT,
            embedding BLOB,
            cluster_id INTEGER,
            identity_id INTEGER,
            confidence REAL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            label TEXT
        );
        
        CREATE TABLE IF NOT EXISTS clip_embeddings (
            image_id INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        
        CREATE TABLE IF NOT EXISTS processing_checkpoint (
            stage_name TEXT,
            image_id INTEGER,
            PRIMARY KEY (stage_name, image_id),
            FOREIGN KEY(image_id) REFERENCES images(id)
        );

        CREATE TABLE IF NOT EXISTS institutional_keywords (
            id INTEGER PRIMARY KEY,
            source_sheet TEXT,
            source_row INTEGER,
            project TEXT,
            original_keyword TEXT NOT NULL,
            normalized_keyword TEXT NOT NULL,
            hard_drive TEXT,
            primary_category TEXT,
            secondary_categories TEXT,
            keyword_type TEXT,
            candidate_visual_class TEXT,
            visual_cues TEXT,
            context_associations TEXT,
            search_expansions TEXT,
            recommended_model_role TEXT,
            evidence_basis TEXT,
            training_use TEXT,
            label_type TEXT NOT NULL,
            priority TEXT,
            inference_note TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ik_category ON institutional_keywords(primary_category);
        CREATE INDEX IF NOT EXISTS idx_ik_label_type ON institutional_keywords(label_type);
        CREATE INDEX IF NOT EXISTS idx_ik_project ON institutional_keywords(project);

        CREATE TABLE IF NOT EXISTS image_institutional_tags (
            image_id INTEGER,
            keyword_id INTEGER,
            similarity_score REAL,
            match_method TEXT,
            PRIMARY KEY (image_id, keyword_id),
            FOREIGN KEY(image_id) REFERENCES images(id),
            FOREIGN KEY(keyword_id) REFERENCES institutional_keywords(id)
        );

        CREATE INDEX IF NOT EXISTS idx_iit_image ON image_institutional_tags(image_id);
        CREATE INDEX IF NOT EXISTS idx_iit_keyword ON image_institutional_tags(keyword_id);
        '''
        
        trigger_captions = '''
        CREATE TRIGGER IF NOT EXISTS captions_ai AFTER INSERT ON caption BEGIN
            INSERT INTO fts_captions(rowid, caption) VALUES (new.image_id, new.caption);
        END;
        CREATE TRIGGER IF NOT EXISTS captions_au AFTER UPDATE ON caption BEGIN
            INSERT INTO fts_captions(fts_captions, rowid, caption) VALUES ('delete', old.image_id, old.caption);
            INSERT INTO fts_captions(rowid, caption) VALUES (new.image_id, new.caption);
        END;
        CREATE TRIGGER IF NOT EXISTS captions_ad AFTER DELETE ON caption BEGIN
            INSERT INTO fts_captions(fts_captions, rowid, caption) VALUES ('delete', old.image_id, old.caption);
        END;
        '''
        
        trigger_tags = '''
        CREATE TRIGGER IF NOT EXISTS tags_ai AFTER INSERT ON tags BEGIN
            INSERT INTO fts_tags(rowid, tag) VALUES (new.id, new.tag);
        END;
        CREATE TRIGGER IF NOT EXISTS tags_au AFTER UPDATE ON tags BEGIN
            INSERT INTO fts_tags(fts_tags, rowid, tag) VALUES ('delete', old.id, old.tag);
            INSERT INTO fts_tags(rowid, tag) VALUES (new.id, new.tag);
        END;
        CREATE TRIGGER IF NOT EXISTS tags_ad AFTER DELETE ON tags BEGIN
            INSERT INTO fts_tags(fts_tags, rowid, tag) VALUES ('delete', old.id, old.tag);
        END;
        '''
        
        with self.get_connection() as conn:
            conn.executescript(schema)
            conn.executescript(trigger_captions)
            conn.executescript(trigger_tags)
            
    def insert_image(self, file_path: str, metadata_dict: Dict[str, Any]) -> int:
        """Inserts an image into the database or updates it if it exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO images (file_path, width, height, format, mode, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    width=excluded.width,
                    height=excluded.height,
                    format=excluded.format,
                    mode=excluded.mode,
                    file_size=excluded.file_size
            ''', (
                file_path,
                metadata_dict.get('width'),
                metadata_dict.get('height'),
                metadata_dict.get('format'),
                metadata_dict.get('mode'),
                metadata_dict.get('file_size')
            ))
            cursor.execute("SELECT id FROM images WHERE file_path=?", (file_path,))
            return cursor.fetchone()['id']

    def insert_tags(self, image_id: int, tags_with_source_and_confidence: List[Tuple[str, str, float]]):
        """Inserts multiple tags for a given image ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO tags (image_id, tag, source, confidence)
                VALUES (?, ?, ?, ?)
            ''', [(image_id, tag, source, conf) for tag, source, conf in tags_with_source_and_confidence])

    def insert_caption(self, image_id: int, caption: str):
        """Inserts or updates an image caption."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO caption (image_id, caption)
                VALUES (?, ?)
                ON CONFLICT(image_id) DO UPDATE SET caption=excluded.caption
            ''', (image_id, caption))

    def insert_face(self, image_id: int, bbox: str, embedding_bytes: bytes, cluster_id: Optional[int] = None, identity_id: Optional[int] = None, confidence: Optional[float] = None) -> int:
        """Inserts a face record and returns its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO faces (image_id, bbox, embedding, cluster_id, identity_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (image_id, bbox, embedding_bytes, cluster_id, identity_id, confidence))
            return cursor.lastrowid

    def update_face_cluster(self, face_id: int, cluster_id: int):
        """Updates the cluster ID for a specific face."""
        with self.get_connection() as conn:
            conn.execute('UPDATE faces SET cluster_id=? WHERE id=?', (cluster_id, face_id))

    def insert_identity(self, name: str, label: str) -> int:
        """Inserts an identity record and returns its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO identities (name, label) VALUES (?, ?)', (name, label))
            return cursor.lastrowid

    def link_face_to_identity(self, face_id: int, identity_id: int):
        """Links a specific face to an identity."""
        with self.get_connection() as conn:
            conn.execute('UPDATE faces SET identity_id=? WHERE id=?', (identity_id, face_id))

    def insert_clip_embedding(self, image_id: int, embedding_bytes: bytes):
        """Inserts or updates a CLIP embedding for a given image ID."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO clip_embeddings (image_id, embedding)
                VALUES (?, ?)
                ON CONFLICT(image_id) DO UPDATE SET embedding=excluded.embedding
            ''', (image_id, embedding_bytes))

    def get_processed_ids(self, stage_name: str) -> List[int]:
        """Gets all image IDs processed in a specific pipeline stage."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT image_id FROM processing_checkpoint WHERE stage_name=?', (stage_name,))
            return [row['image_id'] for row in cursor.fetchall()]

    def mark_processed(self, stage_name: str, image_ids: List[int]):
        """Marks a batch of image IDs as processed in a specific pipeline stage."""
        with self.get_connection() as conn:
            conn.executemany('''
                INSERT OR IGNORE INTO processing_checkpoint (stage_name, image_id)
                VALUES (?, ?)
            ''', [(stage_name, img_id) for img_id in image_ids])

    def get_all_images(self) -> List[Dict[str, Any]]:
        """Retrieves all images."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM images')
            return [dict(row) for row in cursor.fetchall()]

    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a specific image by its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM images WHERE id=?', (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def search_fts(self, query: str, limit: int = 50) -> List[int]:
        """Searches captions and tags using FTS5."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
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
            ''', (query, query, limit))
            return [row['image_id'] for row in cursor.fetchall()]

    def get_all_face_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all non-null face embeddings."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT id, embedding FROM faces WHERE embedding IS NOT NULL')
            return [dict(row) for row in cursor.fetchall()]

    def get_faces_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Retrieves all faces belonging to a specific cluster."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM faces WHERE cluster_id=?', (cluster_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_image_tags(self, image_id: int) -> List[Dict[str, Any]]:
        """Retrieves all tags for a given image ID."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM tags WHERE image_id=?', (image_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_image_faces(self, image_id: int) -> List[Dict[str, Any]]:
        """Retrieves all faces associated with a given image ID."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM faces WHERE image_id=?', (image_id,))
            return [dict(row) for row in cursor.fetchall()]

    def insert_institutional_keywords(self, entries: List[Any]):
        """Inserts or replaces institutional taxonomy keywords into database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO institutional_keywords (
                    id, source_sheet, source_row, project, original_keyword,
                    normalized_keyword, hard_drive, primary_category, secondary_categories,
                    keyword_type, candidate_visual_class, visual_cues, context_associations,
                    search_expansions, recommended_model_role, evidence_basis, training_use,
                    label_type, priority, inference_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (
                    e.keyword_id,
                    getattr(e, 'source_sheet', ''),
                    getattr(e, 'source_row', 0),
                    getattr(e, 'project', ''),
                    e.original_keyword,
                    e.normalized_keyword,
                    getattr(e, 'hard_drive', None),
                    getattr(e, 'primary_category', ''),
                    '; '.join(getattr(e, 'secondary_categories', [])),
                    getattr(e, 'keyword_type', ''),
                    getattr(e, 'candidate_visual_class', None),
                    '; '.join(getattr(e, 'visual_cues', [])),
                    '; '.join(getattr(e, 'context_associations', [])),
                    '; '.join(getattr(e, 'search_expansions', [])),
                    getattr(e, 'recommended_model_role', ''),
                    getattr(e, 'evidence_basis', ''),
                    getattr(e, 'training_use', ''),
                    getattr(e, 'label_type', 'context'),
                    getattr(e, 'priority', 'High'),
                    getattr(e, 'inference_note', None)
                ) for e in entries
            ])

    def insert_image_institutional_tags(self, tags: List[Tuple[int, int, float, str]]):
        """
        Inserts image-to-institutional-keyword match associations.
        tags: list of (image_id, keyword_id, similarity_score, match_method)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO image_institutional_tags (
                    image_id, keyword_id, similarity_score, match_method
                ) VALUES (?, ?, ?, ?)
            ''', tags)

    def get_all_institutional_keywords(self) -> List[Dict[str, Any]]:
        """Retrieves all institutional keywords."""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM institutional_keywords ORDER BY id')
            return [dict(row) for row in cursor.fetchall()]

    def get_taxonomy_categories(self) -> List[str]:
        """Retrieves distinct non-empty primary categories."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT primary_category FROM institutional_keywords 
                WHERE primary_category IS NOT NULL AND primary_category != ''
                ORDER BY primary_category
            ''')
            return [row['primary_category'] for row in cursor.fetchall()]

    def get_taxonomy_projects(self) -> List[str]:
        """Retrieves distinct non-empty projects/events."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT project FROM institutional_keywords 
                WHERE project IS NOT NULL AND project != ''
                ORDER BY project
            ''')
            return [row['project'] for row in cursor.fetchall()]

    def get_image_institutional_tags(self, image_id: int) -> List[Dict[str, Any]]:
        """Retrieves institutional keyword tags linked to an image."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT iit.keyword_id, iit.similarity_score, iit.match_method,
                       ik.normalized_keyword, ik.primary_category, ik.label_type,
                       ik.project, ik.priority
                FROM image_institutional_tags iit
                JOIN institutional_keywords ik ON iit.keyword_id = ik.id
                WHERE iit.image_id = ?
                ORDER BY iit.similarity_score DESC
            ''', (image_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_taxonomy_filter(
        self,
        category: Optional[str] = None,
        project: Optional[str] = None,
        label_type: Optional[str] = None
    ) -> List[int]:
        """Filters image IDs based on institutional taxonomy criteria."""
        clauses = []
        params = []
        if category:
            clauses.append("ik.primary_category = ?")
            params.append(category)
        if project:
            clauses.append("ik.project = ?")
            params.append(project)
        if label_type:
            clauses.append("ik.label_type = ?")
            params.append(label_type)

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f'''
            SELECT DISTINCT iit.image_id
            FROM image_institutional_tags iit
            JOIN institutional_keywords ik ON iit.keyword_id = ik.id
            {where_sql}
        '''
        with self.get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            return [row['image_id'] for row in cursor.fetchall()]
