"""Generate expected EmbeddingGemma-2 text embeddings using HuggingFace.

Run with:
    .venv/bin/python3 tests/oracle/run_embeddinggemma2_text_oracle.py \
        --input "Hello world" \
        --model models/embeddinggemma-2-safetensors \
        --output tests/oracle/embeddinggemma2_text_oracle.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="text to embed")
    p.add_argument("--model", default="models/embeddinggemma-2-safetensors")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype="auto")
    model.eval()
    inputs = tokenizer(args.input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding_list = embedding[0].cpu().float().tolist()
    # Also save token ids for the Rust side to verify exact same input.
    out = {
        "text": args.input,
        "token_ids": inputs["input_ids"][0].cpu().tolist(),
        "embedding": embedding_list,
        "embedding_dim": len(embedding_list),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}: {len(embedding_list)}-dim embedding", file=sys.stderr)


if __name__ == "__main__":
    main()
