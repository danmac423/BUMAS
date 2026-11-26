"""BUMAS-mini: a small LLaVA-style VLM (Qwen3-0.6B-Base + frozen SigLIP2-B/16-224 + pixel-shuffle MLP connector).

Shared by training, attacks and the student release (the release ships this file as `model/openbumas.py`).
An image becomes 49 tokens: <|vision_start|> + 49 x <|image_pad|> + <|vision_end|>; the connector output replaces the
embeddings of the <|image_pad|> positions.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

VISION_START, VISION_END, IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
N_IMG_TOKENS = 49
IMAGE_SIZE = 224


def load_image(path) -> torch.Tensor:
    """SigLIP preprocessing: RGB, 224x224 bicubic, scale to [-1, 1]."""
    im = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - 0.5) / 0.5


class BUMASMini(nn.Module):
    def __init__(self, llm, vision, connector, tokenizer):
        super().__init__()
        self.llm, self.vision, self.connector, self.tok = llm, vision, connector, tokenizer
        self.image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        for p in self.vision.parameters():
            p.requires_grad_(False)

    # ---- construction / IO ----
    @classmethod
    def from_components(cls, llm_path, vision_path, dtype=torch.float32):
        tok = AutoTokenizer.from_pretrained(llm_path)
        llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype)
        vision = SiglipVisionModel.from_pretrained(vision_path, dtype=torch.bfloat16)
        d_v, d_l = vision.config.hidden_size, llm.config.hidden_size
        connector = nn.Sequential(nn.Linear(4 * d_v, 2 * d_l), nn.GELU(), nn.Linear(2 * d_l, d_l)).to(dtype)
        return cls(llm, vision, connector, tok)

    def save(self, out_dir):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(out / "llm", safe_serialization=True)
        self.tok.save_pretrained(out / "llm")
        self.vision.save_pretrained(out / "vision", safe_serialization=True)
        torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in self.connector.state_dict().items()}, out / "connector.pt")
        (out / "openbumas.json").write_text(json.dumps({"n_img_tokens": N_IMG_TOKENS, "image_size": IMAGE_SIZE}))

    @classmethod
    def load(cls, ckpt_dir, dtype=torch.float32):
        ck = Path(ckpt_dir)
        tok = AutoTokenizer.from_pretrained(ck / "llm")
        llm = AutoModelForCausalLM.from_pretrained(ck / "llm", dtype=dtype)
        vision = SiglipVisionModel.from_pretrained(ck / "vision", dtype=torch.bfloat16)
        d_v, d_l = vision.config.hidden_size, llm.config.hidden_size
        connector = nn.Sequential(nn.Linear(4 * d_v, 2 * d_l), nn.GELU(), nn.Linear(2 * d_l, d_l))
        connector.load_state_dict({k: v.float() for k, v in torch.load(ck / "connector.pt").items()})
        return cls(llm, vision, connector.to(dtype), tok)

    # ---- forward ----
    def encode_images(self, pixel_values):
        with torch.no_grad():
            h = self.vision(pixel_values=pixel_values.to(torch.bfloat16)).last_hidden_state  # (B, 196, 768)
        b, n, c = h.shape
        s = int(n ** 0.5)
        h = h.view(b, s // 2, 2, s // 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(b, (s // 2) ** 2, 4 * c)
        return self.connector(h.to(self.connector[0].weight.dtype))  # (B, 49, d_llm)

    def forward(self, input_ids, attention_mask, labels=None, pixel_values=None):
        emb = self.llm.get_input_embeddings()(input_ids)
        if pixel_values is not None and pixel_values.numel():
            img = self.encode_images(pixel_values).to(emb.dtype).reshape(-1, emb.shape[-1])
            mask = input_ids == self.image_pad_id
            assert mask.sum() == img.shape[0], (mask.sum().item(), img.shape)
            emb = emb.masked_scatter(mask.unsqueeze(-1), img)
        if labels is None:
            return self.llm(inputs_embeds=emb, attention_mask=attention_mask)
        # Training: run the lm_head only where a label exists (saves the 152k-vocab logits at every other position).
        h = self.llm.model(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        if not sel.any():  # e.g. a chat whose assistant turn was cut off by max_len: nothing to learn here
            return _Out(h.sum() * 0.0)
        logits = self.llm.lm_head(h[:, :-1][sel])
        loss = nn.functional.cross_entropy(logits.float(), tgt[sel])
        return _Out(loss)


class _Out:
    def __init__(self, loss):
        self.loss = loss


# ---- formatting: every training stage and every attack uses these, so the formats always match ----
def image_block():
    return VISION_START + IMAGE_PAD * N_IMG_TOKENS + VISION_END


def encode_segments(tok, segments, max_len):
    """segments: list of (text, train: bool). Returns ids and labels (-100 where not trained)."""
    ids, labels = [], []
    for text, train in segments:
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if train else [-100] * len(t)
    return ids[:max_len], labels[:max_len]


def caption_example(tok, caption, max_len=256):
    """Image-caption pretraining (S2/S3): image block, then the caption (trained) and EOT."""
    return encode_segments(tok, [(image_block(), False), (caption.strip() + EOT, True)], max_len)


def chat_example(tok, messages, has_image=False, max_len=2048):
    """ChatML; loss only on assistant turns (content + <|im_end|>). If has_image, the image block opens the first user turn."""
    segs, first_user = [], True
    for m in messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        if role == "user":
            body = (image_block() + "\n" if has_image and first_user else "") + m["content"]
            first_user = False
            segs.append((f"{IM_START}user\n{body}{IM_END}\n", False))
        else:
            segs.append((f"{IM_START}assistant\n", False))
            segs.append((m["content"] + IM_END, True))
            segs.append(("\n", False))
    return encode_segments(tok, segs, max_len)


def text_example(tok, text, max_len=2048):
    return encode_segments(tok, [(text.strip() + EOT, True)], max_len)
