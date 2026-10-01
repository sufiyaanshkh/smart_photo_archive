"""Search result fusion methods."""

import logging
from typing import List, Tuple, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists: List of ranked lists, where each list contains (image_id, score) tuples.
        k: The constant k in RRF formula (default 60).
        
    Returns:
        Fused ranked list of (image_id, fused_score).
    """
    if not ranked_lists:
        return []
        
    rrf_scores: Dict[int, float] = defaultdict(float)
    
    for ranked_list in ranked_lists:
        # Sort by score descending to ensure correct ranking
        sorted_list = sorted(ranked_list, key=lambda x: x[1], reverse=True)
        for rank, (image_id, _) in enumerate(sorted_list, 1):
            rrf_scores[image_id] += 1.0 / (k + rank)
            
    # Sort by RRF score descending
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results

def weighted_fusion(ranked_lists: List[List[Tuple[int, float]]], weights: List[float]) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using weighted score aggregation.
    
    Args:
        ranked_lists: List of ranked lists, each containing (image_id, score) tuples.
        weights: List of weights corresponding to each ranked list.
        
    Returns:
        Fused ranked list of (image_id, fused_score).
    """
    if not ranked_lists or not weights:
        return []
        
    if len(ranked_lists) != len(weights):
        logger.error(f"Length mismatch: {len(ranked_lists)} lists, {len(weights)} weights")
        raise ValueError("Number of ranked lists must match number of weights")
        
    fused_scores: Dict[int, float] = defaultdict(float)
    
    for ranked_list, weight in zip(ranked_lists, weights):
        normalized_list = normalize_scores(ranked_list)
        for image_id, score in normalized_list:
            fused_scores[image_id] += score * weight
            
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results

def normalize_scores(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Normalize scores using min-max normalization.
    
    Args:
        results: List of (image_id, score) tuples.
        
    Returns:
        List of (image_id, normalized_score) tuples.
    """
    if not results:
        return []
        
    if len(results) == 1:
        return [(results[0][0], 1.0)]
        
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [(img_id, 1.0) for img_id, _ in results]
        
    normalized = []
    for img_id, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((img_id, norm_score))
        
    return normalized
