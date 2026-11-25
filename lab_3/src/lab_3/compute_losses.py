"""Stage 1 (GPU, run once): per-token losses of BUMAS-mini and of the reference model for every record.

Both models score exactly the same target tokens:

* text records: ``text_example`` (as in training stage S1); every token that has a left context is scored,
* photo records: BUMAS-mini sees the image block followed by the caption, the reference model sees the caption
  preceded by ``<|endoftext|>``; only the caption tokens are scored.

Partial results of each model are saved while running, and the next run resumes from them. After changing how records
are tokenised or scored, delete ``artifacts/partial_*.pkl``.
"""

import gc
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab_3.checkpoint import Checkpoint, atomic_pickle
from lab_3.model.openbumas import EOT, BUMASMini, caption_example, load_image, text_example
from lab_3.records import DATA_DIR, LOSSES_PATH, MODEL_DIR, Record, load_jsonl

REFERENCE_MODEL = "Qwen/Qwen3-0.6B-Base"
# Same precision for both models, so that the arithmetic cannot shift the loss difference between them.
DTYPE = torch.bfloat16
# Upper bound on batch_size * padded_length.
MAX_BATCH_TOKENS = 4096
MAX_BATCH_SIZE = 64
# Number of positions sent through the lm_head at once (the vocabulary has about 152k entries).
HEAD_CHUNK = 512


@dataclass
class Example:
    """One tokenised record."""

    record_id: str
    ids: list[int]
    labels: list[int]
    image: str | None = None


def build_examples(tok, records: list[Record], reference: bool) -> list[Example]:
    """Tokenise records the way each of the two models should see them.

    Args:
        tok: Tokenizer shared by both models.
        records: Records to tokenise.
        reference: Build inputs for the text-only reference model instead of BUMAS-mini.

    Returns:
        One example per record, in the same order.
    """
    eot_id = tok.convert_tokens_to_ids(EOT)
    examples = []
    for r in records:
        if r["kind"] == "text":
            ids, labels = text_example(tok, r["text"])
            examples.append(Example(r["id"], ids, labels))
            continue
        ids, labels = caption_example(tok, r["caption"])
        if reference:
            # Drop the image block; EOT gives the first caption token the context it has in BUMAS-mini.
            caption_ids = [i for i, label in zip(ids, labels, strict=True) if label != -100]
            examples.append(Example(r["id"], [eot_id, *caption_ids], [-100, *caption_ids]))
        else:
            examples.append(Example(r["id"], ids, labels, r["image"]))
    return examples


def batches(examples: list[Example]) -> Iterator[list[Example]]:
    """Group examples of similar length (and the same modality) to limit padding.

    Args:
        examples: Examples in any order.

    Yields:
        Batches within the ``MAX_BATCH_TOKENS`` and ``MAX_BATCH_SIZE`` limits.
    """
    batch: list[Example] = []
    for ex in sorted(examples, key=lambda e: (e.image is not None, len(e.ids))):
        full = batch and (
            len(batch) == MAX_BATCH_SIZE
            or (len(batch) + 1) * len(ex.ids) > MAX_BATCH_TOKENS
            or (batch[0].image is None) != (ex.image is None)
        )
        if full:
            yield batch
            batch = []
        batch.append(ex)
    if batch:
        yield batch


@torch.no_grad()
def token_losses(llm, batch: list[Example], pad_id: int, bumas: BUMASMini | None = None) -> list[np.ndarray]:
    """Cross-entropy of every labelled token in a batch.

    Args:
        llm: Causal language model.
        batch: Examples to score; either all with an image or all without.
        pad_id: Token used for right padding.
        bumas: Full BUMAS-mini, needed to encode images.

    Returns:
        One float32 array of per-token losses per example.
    """
    device = llm.device
    length = max(len(e.ids) for e in batch)
    input_ids = torch.tensor([e.ids + [pad_id] * (length - len(e.ids)) for e in batch], device=device)
    labels = torch.tensor([e.labels + [-100] * (length - len(e.ids)) for e in batch], device=device)
    attention_mask = torch.tensor([[1] * len(e.ids) + [0] * (length - len(e.ids)) for e in batch], device=device)

    emb = llm.get_input_embeddings()(input_ids)
    if bumas is not None and batch[0].image is not None:
        pixels = torch.stack([load_image(DATA_DIR / e.image) for e in batch if e.image is not None]).to(device)
        img = bumas.encode_images(pixels).to(emb.dtype).reshape(-1, emb.shape[-1])
        mask = input_ids == bumas.image_pad_id
        emb = emb.masked_scatter(mask.unsqueeze(-1), img)

    hidden = llm.model(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False).last_hidden_state
    targets = labels[:, 1:]
    selected = targets != -100
    states, wanted = hidden[:, :-1][selected], targets[selected]
    losses = torch.cat(
        [
            torch.nn.functional.cross_entropy(llm.lm_head(s).float(), t, reduction="none")
            for s, t in zip(states.split(HEAD_CHUNK), wanted.split(HEAD_CHUNK), strict=True)
        ]
    )
    counts = selected.sum(dim=1).tolist()
    return [part.numpy().astype(np.float32) for part in losses.cpu().split(counts)]


def open_checkpoint(name: str, model: str) -> Checkpoint:
    """Checkpoint of the partial losses of one model."""
    return Checkpoint(LOSSES_PATH.with_name(f"partial_{name}.pkl"), {"model": model, "dtype": str(DTYPE)})


def score_all(llm, examples: list[Example], pad_id: int, desc: str, ckpt: Checkpoint, bumas: BUMASMini | None = None):
    """Compute the losses of the examples that are not in the checkpoint yet.

    The checkpoint is saved periodically and also when the loop is interrupted.
    """
    pending = [e for e in examples if e.record_id not in ckpt.results]
    try:
        with tqdm(total=len(examples), initial=len(examples) - len(pending), desc=desc) as bar:
            for batch in batches(pending):
                losses = token_losses(llm, batch, pad_id, bumas)
                ckpt.update({ex.record_id: loss for ex, loss in zip(batch, losses, strict=True)})
                bar.update(len(batch))
    finally:
        ckpt.save()


def main() -> None:
    """Compute the losses of both models and write them to the cache."""
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    records = load_jsonl(DATA_DIR / "known.jsonl") + load_jsonl(DATA_DIR / "test.jsonl")

    record_ids = {r["id"] for r in records}
    target_ckpt = open_checkpoint("bumas", "BUMAS-mini")
    reference_ckpt = open_checkpoint("reference", REFERENCE_MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / "llm")
    if tok is None:
        raise RuntimeError(f"No tokenizer in {MODEL_DIR}")
    pad_id = tok.convert_tokens_to_ids(EOT)
    if not isinstance(pad_id, int):
        raise TypeError(f"{EOT} must be a single token")

    # A model is loaded only if some of its records are still missing.
    if record_ids - target_ckpt.results.keys():
        bumas = BUMASMini.load(MODEL_DIR, dtype=DTYPE).eval().to("cuda")
        examples = build_examples(tok, records, reference=False)
        score_all(bumas.llm, examples, pad_id, "BUMAS-mini", target_ckpt, bumas)
        del bumas
        gc.collect()
        torch.cuda.empty_cache()

    if record_ids - reference_ckpt.results.keys():
        ref_llm = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL, dtype=DTYPE).eval().to("cuda")
        examples = build_examples(tok, records, reference=True)
        score_all(ref_llm, examples, pad_id, "reference", reference_ckpt)

    target, reference = target_ckpt.results, reference_ckpt.results
    cache = {}
    for r in records:
        b, q = target[r["id"]], reference[r["id"]]
        if len(b) != len(q) or len(b) == 0:
            raise RuntimeError(f"Token mismatch for {r['id']}: {len(b)} vs {len(q)}")
        cache[r["id"]] = {
            "bumas_tok": b,
            "ref_tok": q,
            "bumas_mean": float(b.mean()),
            "ref_mean": float(q.mean()),
            "n_tokens": len(b),
        }
    atomic_pickle(LOSSES_PATH, cache)
    for ckpt in (target_ckpt, reference_ckpt):
        ckpt.path.unlink()
    print(f"Saved losses of {len(cache)} records to {LOSSES_PATH}")


if __name__ == "__main__":
    main()
