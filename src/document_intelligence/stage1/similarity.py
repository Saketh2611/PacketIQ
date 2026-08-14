"""Reusable similarity functions."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from document_intelligence.ingestion.page_extractor import PageRepresentation, TextBlock


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalized_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def text_length_ratio(len_a: int, len_b: int) -> float:
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def layout_similarity(page_a: PageRepresentation, page_b: PageRepresentation) -> float:
    block_ratio = text_length_ratio(page_a.block_count, page_b.block_count)
    dim_ratio = text_length_ratio(int(page_a.width), int(page_b.width))
    dim_ratio = (dim_ratio + text_length_ratio(int(page_a.height), int(page_b.height))) / 2
    return (block_ratio + dim_ratio) / 2


def block_position_similarity(blocks_a: list[TextBlock], blocks_b: list[TextBlock]) -> float:
    if not blocks_a or not blocks_b:
        return 0.0

    def profile(blocks: list[TextBlock]) -> list[float]:
        if not blocks:
            return [0.0] * 5
        ys = [b.y0 for b in blocks]
        xs = [b.x0 for b in blocks]
        return [
            min(ys) / 1000,
            max(ys) / 1000,
            np.mean(xs) / 1000,
            len(blocks) / 50,
            sum(1 for b in blocks if b.is_heading) / max(len(blocks), 1),
        ]

    pa = np.array(profile(blocks_a))
    pb = np.array(profile(blocks_b))
    diff = np.abs(pa - pb)
    return float(max(0.0, 1.0 - np.mean(diff)))


def heading_style_similarity(page_a: PageRepresentation, page_b: PageRepresentation) -> float:
    ha = sum(1 for b in page_a.blocks if b.is_heading)
    hb = sum(1 for b in page_b.blocks if b.is_heading)
    return text_length_ratio(ha, hb)


def jaccard_entities(text_a: str, text_b: str) -> float:
    pattern = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b|\b\d{4,}\b|\$[\d,.]+"
    ents_a = set(re.findall(pattern, text_a))
    ents_b = set(re.findall(pattern, text_b))
    return normalized_overlap(ents_a, ents_b)


def structural_similarity(page_a: PageRepresentation, page_b: PageRepresentation) -> float:
    scores = [
        text_length_ratio(page_a.text_length, page_b.text_length),
        text_length_ratio(page_a.image_count, page_b.image_count),
        heading_style_similarity(page_a, page_b),
    ]
    return float(np.mean(scores))
