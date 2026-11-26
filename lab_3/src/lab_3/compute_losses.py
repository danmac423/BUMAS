"""Step 1: the loss of every token of every record under BUMAS-mini and under the reference model.

The reference model is BUMAS-mini with its LLM replaced by the public Qwen3-0.6B-Base, which OpenBUMAS started from.
The vision encoder, connector and tokenizer are the same, so both models get exactly the same input.

Run from the project root:  uv run python -m lab_3.compute_losses
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lab_3.model.openbumas import BUMASMini, caption_example, load_image, text_example

DATA = Path("data")
MODEL_DIR = Path(__file__).parent / "model"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
OUTPUT = Path("outputs/losses.pt")
DTYPE = torch.bfloat16  # in float32 the two models do not fit in 10 GB of GPU memory
CHUNK = 256


def read_records(name: str) -> list[dict]:
    """Read a .jsonl file from the data directory."""
    with open(DATA / name, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@torch.no_grad()
def token_losses(model: BUMASMini, record: dict) -> torch.Tensor:
    """Return -log p(token) for every token the model was trained to predict, formatted exactly as in training.

    For texts these are all tokens but the first one, for photos only the tokens of the caption.
    """
    if record["kind"] == "photo":
        ids, labels = caption_example(model.tok, record["caption"])
        image = load_image(DATA / record["image"])
    else:
        ids, labels = text_example(model.tok, record["text"])
        image = None

    # The same steps as in BUMASMini.forward, but the logits are computed chunk by chunk:
    # for 2048 tokens and a vocabulary of 150k they do not fit in GPU memory all at once.
    device = model.llm.device
    input_ids = torch.tensor([ids], device=device)
    embeddings = model.llm.get_input_embeddings()(input_ids)
    if image is not None:  # the image vectors replace the <|image_pad|> tokens
        image_vectors = model.encode_images(image[None].to(device)).to(embeddings.dtype)
        is_image = (input_ids == model.image_pad_id)[..., None]
        embeddings = embeddings.masked_scatter(is_image, image_vectors.reshape(-1, embeddings.shape[-1]))
    hidden = model.llm.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state[0, :-1]

    targets = torch.tensor(labels[1:], device=device)  # the output at position t predicts token t + 1
    trained = targets != -100
    losses = [
        torch.nn.functional.cross_entropy(model.llm.lm_head(h).float(), t, reduction="none").cpu()
        for h, t in zip(hidden[trained].split(CHUNK), targets[trained].split(CHUNK), strict=True)
    ]
    return torch.cat(losses)


def main() -> None:
    """Compute and save the losses of all known and test records."""
    bumas = BUMASMini.load(MODEL_DIR, dtype=DTYPE).cuda().eval()
    reference = BUMASMini.load(MODEL_DIR, dtype=DTYPE)
    reference.llm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE)
    reference = reference.cuda().eval()

    records = read_records("known.jsonl") + read_records("test.jsonl")
    results = {
        record["id"]: {"bumas": token_losses(bumas, record), "reference": token_losses(reference, record)}
        for record in tqdm(records, mininterval=30)
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    torch.save(results, OUTPUT)


if __name__ == "__main__":
    main()
