# 🎙️ Voice2Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)

An **AI-powered voice-to-visual system** that transforms spoken commands into stunning **images** and **animated videos** using state-of-the-art deep learning models.

---

## 🚀 Getting Started – Run Free on Google Colab

Get up and running in minutes without any local setup required. Click the badge above or follow the link below:

👉 **[Launch in Google Colab](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)**

### Quick Setup Guide:

1. **Enable GPU Acceleration**: Navigate to **Runtime** → **Change runtime type** → Select **T4 GPU** → Click **Save**
2. **Step 1**: Click the Play (▶) button to install MOSS and required AI libraries (approximately 30 seconds)
3. **Step 2**: Click the Play (▶) button to load the models and launch the Gradio Web App
4. **Access Your App**: Open the generated public web URL (`https://xxxx.gradio.live`) in your browser and start creating!

---

## 🧠 AI Models & Architecture

Our system leverages cutting-edge models to deliver exceptional results:

| Component | Model / Technology |
| :--- | :--- |
| **Speech-to-Text** | `OpenMOSS-Team/MOSS-Transcribe-Diarize` (0.9B CausalLM) |
| **Image Generation** | `dreamlike-art/dreamlike-diffusion-1.0` (Stable Diffusion – Watermark Free) |
| **Video Generation** | `AnimateDiffPipeline` with `AnimateLCM` + `emilianJR/epiCRealism` (16-Frame AI Video) |
| **User Interface** | Gradio Web App (`demo.launch(share=True)`) |

---

## 📁 Repository Structure

Here's what you'll find in this project:

- **`run_on_colab.ipynb`** – Ready-to-run Colab notebook with full 2-step setup and integrated Gradio interface
- **`voice to image and video model1.ipynb`** – Comprehensive project notebook featuring complete AI pipelines and workflows
- **`app.py`** – Standalone backend server optimized for GPU hosting or local execution with NVIDIA CUDA
- **`templates/` & `static/`** – Web frontend assets including HTML, CSS, and JavaScript
