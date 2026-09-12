---
title: Voice2Vision
emoji: 🎙️
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: "4.44.0"
app_file: hf_app.py
pinned: false
license: mit
short_description: Speak a scene — watch it come alive
---

# 🎙️ Voice2Vision

**AI-powered voice-to-visual system** that converts spoken commands into images and animated videos.

## How it works

1. **Record** your voice or upload an audio file
2. **Transcribe** — Whisper converts speech to text
3. **Generate Image** — Pollinations.ai creates an image from your words
4. **Generate Video** — An animated GIF is built from multiple generated frames

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | OpenAI Whisper (base model) |
| Image Generation | Pollinations.ai free API |
| Animated Video | PIL GIF stitching from Pollinations frames |
| UI | Gradio |

## Run locally

```bash
git clone https://github.com/Swaraj-Palai/Voice2Vision.git
cd Voice2Vision
pip install -r requirements_hf.txt
python hf_app.py
```

## Original notebook

The original prototype (`voice to image and video model1.ipynb`) was built for
Google Colab with a T4 GPU using Stable Diffusion + AnimateDiff.
This deployment uses free cloud APIs so it runs on any machine — no GPU needed.
