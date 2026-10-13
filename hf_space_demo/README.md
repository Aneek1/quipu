---
title: Quipu chat
emoji: 🧶
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat with a 1B MoE trained from scratch for under $20
models:
- AneekC/quipu-moe-1B-A149M-chat-distill
---

Chat demo for [quipu-moe-1B-A149M-chat-distill](https://huggingface.co/AneekC/quipu-moe-1B-A149M-chat-distill),
a small mixture-of-experts model trained from scratch. It runs the model repo's own
standalone loader on a free CPU machine. Small and often wrong; see the model card for
what it can and can't do.
