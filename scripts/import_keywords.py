#!/usr/bin/env python3
"""
Institutional Keyword & Taxonomy Importer.

Imports institutional AI Tagging Taxonomy from Excel into SQLite and
pre-computes multi-prompt CLIP text embeddings for fast pipeline matching.

Usage:
    python -m scripts.import_keywords --excel-path data/institutional_taxonomy.xlsx
    python scripts/import_keywords.py --excel-path data/institutional_taxonomy.xlsx --config config.yaml
"""

import os
import sys
import argparse
import logging
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from storage.database import DatabaseManager
from pipeline.institutional_taxonomy import InstitutionalTaxonomy
from pipeline.clip_embedder import CLIPEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s")
logger = logging.getLogger("import_keywords")


def parse_args():
    parser = argparse.ArgumentParser(description="Import institutional taxonomy from Excel and compute embeddings")
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help="Path to institutional taxonomy Excel file (default from config: data/institutional_taxonomy.xlsx)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for embeddings (.npy file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for CLIP text embedding (cuda or cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    excel_path = args.excel_path or config.paths.taxonomy_excel_path
    output_path = args.output or config.paths.keyword_embeddings_path
    db_path = config.paths.database_path

    if not os.path.exists(excel_path):
        logger.error(f"Taxonomy Excel file not found at: {excel_path}")
        sys.exit(1)

    logger.info(f"Loading institutional taxonomy from {excel_path}...")
    taxonomy = InstitutionalTaxonomy.from_excel(
        excel_path=excel_path,
        main_sheet=config.taxonomy.main_sheet,
        yolo_sheet=config.taxonomy.yolo_sheet,
    )

    if len(taxonomy) == 0:
        logger.error("No taxonomy entries loaded from Excel file!")
        sys.exit(1)

    logger.info(f"Successfully loaded {len(taxonomy)} taxonomy entries and {len(taxonomy.yolo_classes)} YOLO classes.")

    # 1. Store taxonomy in SQLite database
    logger.info(f"Updating database schema and storing taxonomy entries in {db_path}...")
    db = DatabaseManager(db_path)
    db.init_db()
    db.insert_institutional_keywords(taxonomy.entries)
    logger.info("Database updated with institutional keywords.")

    # 2. Compute multi-prompt embeddings using CLIP
    logger.info("Initializing CLIPEmbedder for text embeddings...")
    try:
        clip_embedder = CLIPEmbedder(
            model_name=config.models.clip_model_name,
            pretrained=config.models.clip_pretrained,
            device=args.device,
        )
    except Exception as e:
        logger.warning(f"Could not load open_clip model ({e}). Using mock/fallback 768-d embeddings for development.")
        clip_embedder = None

    all_embeddings = []
    records = []
    keywords_list = []

    logger.info("Computing multi-prompt pooled embeddings for each keyword...")
    for idx, entry in enumerate(taxonomy.entries):
        prompts = entry.get_embedding_prompts()
        if not prompts:
            prompts = [entry.normalized_keyword]

        if clip_embedder is not None:
            prompt_embs = clip_embedder.embed_texts_batch(prompts)
            pooled = np.mean(prompt_embs, axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 1e-6:
                pooled = pooled / norm
        else:
            # Fallback normalized 768-d vector if ML environment doesn't have PyTorch weights ready
            np.random.seed(entry.keyword_id)
            pooled = np.random.randn(768).astype(np.float32)
            pooled = pooled / np.linalg.norm(pooled)

        all_embeddings.append(pooled)
        keywords_list.append(entry.normalized_keyword)
        records.append({
            'id': entry.keyword_id,
            'keyword': entry.normalized_keyword,
            'original': entry.original_keyword,
            'category': entry.primary_category,
            'project': entry.project,
            'label_type': entry.label_type,
            'priority': entry.priority,
            'visual_class': entry.candidate_visual_class,
            'inference_note': entry.inference_note,
        })

    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)

    # 3. Save pre-computed embeddings and metadata dictionary
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    payload = {
        'embeddings': embeddings_matrix,
        'keywords': keywords_list,
        'records': records,
        'categories': taxonomy.get_all_categories(),
        'projects': taxonomy.get_all_projects(),
    }
    np.save(output_path, payload)
    logger.info(f"Saved pre-computed taxonomy embeddings to {output_path} (shape: {embeddings_matrix.shape}).")

    # 4. Print Summary
    print("\n" + "=" * 65)
    print("  INSTITUTIONAL TAXONOMY IMPORT SUMMARY")
    print("=" * 65)
    print(f"  Source Excel        : {excel_path}")
    print(f"  Total Keywords      : {len(taxonomy)}")
    print(f"  Visual YOLO Classes : {len(taxonomy.yolo_classes)}")
    print(f"  Projects Identified : {len(taxonomy.get_all_projects())} {taxonomy.get_all_projects()}")
    print(f"  Primary Categories  : {len(taxonomy.get_all_categories())}")
    print("  Counts by Label Type:")
    for lt in ['object', 'context', 'scene', 'metadata', 'entity_metadata']:
        print(f"    - {lt:16s} : {len(taxonomy.get_by_label_type(lt))}")
    print(f"  Database Updated    : {db_path} (table: institutional_keywords)")
    print(f"  Embeddings Saved    : {output_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
