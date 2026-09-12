# Voice2Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)

Transform your voice into stunning visuals. Voice2Vision is an innovative AI-powered system that converts your spoken ideas directly into high-quality images and captivating animated videos. Whether you're exploring creative possibilities or bringing your imagination to life, this tool makes the process effortless and intuitive.

---

## Run It Now – Completely Free on Google Colab

No installation headaches. No credit card required. Just click and create.

**[Start Creating Now](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)**

### It's as simple as 1-2-3:

1. **Set up your environment**: Open the notebook and switch to **Runtime** → **Change runtime type** → pick **T4 GPU** → **Save**
2. **Load everything up**: Hit play on the first cell to install dependencies (takes about 30 seconds while you grab a coffee)
3. **Start creating**: Run the second cell, wait for the Gradio interface to load, then open the public link and begin transforming your voice into art

That's it. You're ready to create.

---

## The Tech Behind the Magic

We've assembled the best-in-class models to make this work seamlessly:

| What It Does | Technology |
| :--- | :--- |
| **Listens to you** | `OpenMOSS-Team/MOSS-Transcribe-Diarize` — understands your voice with 0.9B parameters of pure listening power |
| **Paints your vision** | `dreamlike-art/dreamlike-diffusion-1.0` — creates watermark-free images from your descriptions |
| **Brings it to life** | `AnimateDiffPipeline` + `AnimateLCM` + `emilianJR/epiCRealism` — generates smooth 16-frame videos that feel real |
| **Shows your work** | Gradio Web App — a clean, simple interface to interact with everything |

---

## What's Inside

**`run_on_colab.ipynb`**  
Your launchpad. Open this notebook in Colab and you're ready to go. Everything is pre-configured, just run the cells in order.

**`voice to image and video model1.ipynb`**  
The complete implementation. Deep dive into the full pipeline here if you want to understand how it all works or customize it.

**`app.py`**  
A standalone server for power users. Host this locally with NVIDIA CUDA support or deploy it on GPU-enabled infrastructure for continuous access.

**`templates/` & `static/`**  
The frontend. HTML, CSS, and JavaScript files that power the web interface you interact with.

---

## How to Use

1. Speak your idea clearly into the microphone
2. Watch as the AI transcribes, interprets, and creates
3. Get back beautiful images and videos ready to share

That's the whole flow. Simple, fast, and surprisingly powerful.
