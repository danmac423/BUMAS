# BUMAS-mini (model card)

A small vision-language assistant (about 0.7B parameters), LLaVA-style.

| Part | Component |
|---|---|
| Language model | initialised from **Qwen/Qwen3-0.6B-Base** (public, Apache-2.0), then trained in stages S1 and S3 |
| Vision encoder | **google/siglip2-base-patch16-224** (public, Apache-2.0), frozen in every stage |
| Connector | pixel-shuffle 2x2 + 2-layer MLP, 49 image tokens per image, trained from scratch (S2 and S3) |

Training stages are described in the task README. Data: a mix of public datasets and web data collected by OpenBUMAS.

Loading:
```python
from openbumas import BUMASMini, load_image, text_example, chat_example, caption_example
model = BUMASMini.load("model").cuda().eval()
```
`openbumas.py` contains the exact input formats used in training (plain text, ChatML chats with loss on assistant turns,
image + caption).
