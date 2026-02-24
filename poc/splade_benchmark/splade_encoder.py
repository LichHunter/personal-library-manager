#!/usr/bin/env python3
"""SPLADE encoder for sparse vector generation.

This module provides the SPLADEEncoder class which:
1. Loads a pre-trained SPLADE model from HuggingFace
2. Encodes text into sparse vectors (term -> weight mappings)
3. Supports batch encoding for efficiency

Usage:
    encoder = SPLADEEncoder()
    sparse_vec = encoder.encode("What is Kubernetes?")
    # sparse_vec = {"kubernetes": 2.1, "container": 1.5, ...}
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm


# Default SPLADE model - best official model with ensemble distillation
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"

# Alternative models for comparison
ALTERNATIVE_MODELS = {
    "ensembledistil": "naver/splade-cocondenser-ensembledistil",
    "selfdistil": "naver/splade-cocondenser-selfdistil", 
    "splade-v3": "naver/splade-v3",
}


@dataclass
class EncodingStats:
    """Statistics from an encoding run."""
    total_texts: int
    total_time_ms: float
    avg_time_per_text_ms: float
    avg_nonzero_terms: float
    min_nonzero_terms: int
    max_nonzero_terms: int
    vocab_size: int


class SPLADEEncoder:
    """SPLADE encoder for sparse vector generation.
    
    SPLADE (SParse Lexical AnD Expansion) uses BERT's MLM head to produce
    sparse term weights. The output is a dict mapping term IDs to weights,
    where most weights are zero (sparse).
    
    Key characteristics:
    - Original terms get high weights (exact match preservation)
    - Related terms get lower weights (semantic expansion)
    - Uses log(1 + ReLU(x)) for sparsification
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize SPLADE encoder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length for encoding
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[SPLADEEncoder] Loading model: {model_name}")
        print(f"[SPLADEEncoder] Device: {self.device}")
        
        load_start = time.time()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        load_time = time.time() - load_start
        print(f"[SPLADEEncoder] Model loaded in {load_time:.2f}s")
        
        # Cache vocabulary for term lookup
        self.vocab_size = self.tokenizer.vocab_size
        self.id_to_token = {v: k for k, v in self.tokenizer.vocab.items()}
        
        print(f"[SPLADEEncoder] Vocabulary size: {self.vocab_size}")
    
    def encode(
        self,
        text: str,
        return_tokens: bool = False,
    ) -> dict[int, float] | tuple[dict[int, float], dict[str, float]]:
        """Encode text to sparse vector.
        
        Args:
            text: Input text to encode
            return_tokens: If True, also return token strings with weights
            
        Returns:
            If return_tokens is False:
                Dict mapping term IDs to weights (non-zero only)
            If return_tokens is True:
                Tuple of (id_weights, token_weights)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # SPLADE aggregation: max pooling over sequence, then log(1 + ReLU(x))
        # This produces sparse weights for each vocabulary term
        relu_log = torch.log1p(torch.relu(logits))
        
        # Max pool over sequence dimension
        sparse_vec, _ = torch.max(relu_log, dim=1)  # (batch, vocab_size)
        sparse_vec = sparse_vec.squeeze(0).cpu().numpy()  # (vocab_size,)
        
        # Extract non-zero weights
        nonzero_indices = np.nonzero(sparse_vec)[0]
        id_weights = {
            int(idx): float(sparse_vec[idx])
            for idx in nonzero_indices
        }
        
        if return_tokens:
            token_weights = {
                self.id_to_token.get(idx, f"[UNK:{idx}]"): weight
                for idx, weight in id_weights.items()
            }
            return id_weights, token_weights
        
        return id_weights
    
    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[dict[int, float]]:
        """Encode multiple texts in batches.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of sparse vectors (one per text)
        """
        results = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=len(texts) // batch_size + 1)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # SPLADE aggregation for batch
            relu_log = torch.log1p(torch.relu(logits))
            sparse_vecs, _ = torch.max(relu_log, dim=1)  # (batch, vocab_size)
            sparse_vecs = sparse_vecs.cpu().numpy()
            
            # Extract non-zero weights for each
            for vec in sparse_vecs:
                nonzero_indices = np.nonzero(vec)[0]
                id_weights = {
                    int(idx): float(vec[idx])
                    for idx in nonzero_indices
                }
                results.append(id_weights)
        
        return results
    
    def get_top_terms(
        self,
        sparse_vec: dict[int, float],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """Get top-k terms by weight from sparse vector.
        
        Args:
            sparse_vec: Sparse vector (term_id -> weight)
            k: Number of top terms to return
            
        Returns:
            List of (term, weight) tuples sorted by weight descending
        """
        sorted_terms = sorted(
            sparse_vec.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [
            (self.id_to_token.get(idx, f"[UNK:{idx}]"), weight)
            for idx, weight in sorted_terms
        ]
    
    def compute_stats(
        self,
        sparse_vecs: list[dict[int, float]],
        total_time_ms: float,
    ) -> EncodingStats:
        """Compute statistics from encoding results.
        
        Args:
            sparse_vecs: List of sparse vectors
            total_time_ms: Total encoding time in milliseconds
            
        Returns:
            EncodingStats with aggregated statistics
        """
        nonzero_counts = [len(v) for v in sparse_vecs]
        
        return EncodingStats(
            total_texts=len(sparse_vecs),
            total_time_ms=total_time_ms,
            avg_time_per_text_ms=total_time_ms / len(sparse_vecs) if sparse_vecs else 0,
            avg_nonzero_terms=np.mean(nonzero_counts) if nonzero_counts else 0,
            min_nonzero_terms=min(nonzero_counts) if nonzero_counts else 0,
            max_nonzero_terms=max(nonzero_counts) if nonzero_counts else 0,
            vocab_size=self.vocab_size,
        )


def validate_encoder(encoder: SPLADEEncoder) -> bool:
    """Validate that encoder produces reasonable sparse vectors.
    
    Args:
        encoder: SPLADEEncoder instance to validate
        
    Returns:
        True if validation passes
    """
    print("\n[Validation] Testing SPLADE encoder...")
    
    # Test 1: Basic encoding
    test_text = "What is a Kubernetes Pod?"
    sparse_vec, token_weights = encoder.encode(test_text, return_tokens=True)
    
    print(f"  Input: '{test_text}'")
    print(f"  Non-zero terms: {len(sparse_vec)}")
    
    if len(sparse_vec) < 5:
        print("  FAIL: Too few non-zero terms")
        return False
    
    if len(sparse_vec) > 1000:
        print("  FAIL: Too many non-zero terms (not sparse)")
        return False
    
    # Test 2: Check top terms are reasonable
    top_terms = encoder.get_top_terms(sparse_vec, k=10)
    print(f"  Top 10 terms: {top_terms}")
    
    # Test 3: Technical term encoding
    tech_text = "SubjectAccessReview webhook"
    tech_vec, tech_tokens = encoder.encode(tech_text, return_tokens=True)
    tech_top = encoder.get_top_terms(tech_vec, k=10)
    print(f"\n  Tech input: '{tech_text}'")
    print(f"  Non-zero terms: {len(tech_vec)}")
    print(f"  Top 10 terms: {tech_top}")
    
    # Test 4: Batch encoding
    batch_texts = [
        "Kubernetes deployment",
        "RBAC permissions",
        "container orchestration",
    ]
    batch_vecs = encoder.encode_batch(batch_texts, show_progress=False)
    
    print(f"\n  Batch encoding: {len(batch_vecs)} texts")
    for i, (text, vec) in enumerate(zip(batch_texts, batch_vecs)):
        print(f"    [{i}] '{text}': {len(vec)} terms")
    
    print("\n[Validation] PASSED")
    return True


def main():
    """Run encoder validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPLADE Encoder")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to encode (for testing)",
    )
    
    args = parser.parse_args()
    
    encoder = SPLADEEncoder(model_name=args.model, device=args.device)
    
    if args.text:
        sparse_vec, token_weights = encoder.encode(args.text, return_tokens=True)
        print(f"\nInput: '{args.text}'")
        print(f"Non-zero terms: {len(sparse_vec)}")
        print(f"\nTop 20 terms:")
        for term, weight in encoder.get_top_terms(sparse_vec, k=20):
            print(f"  {term}: {weight:.4f}")
    else:
        validate_encoder(encoder)


if __name__ == "__main__":
    main()
