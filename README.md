# 🎙️ Voice2Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)

**AI-powered voice-to-visual system** that converts spoken commands into **Images** and **Animated Videos** using authentic deep-learning models.

---

## 🚀 Run 100% Free on Google Colab

Click the badge above or use this link:
👉 **[Open in Google Colab](https://colab.research.google.com/github/Swaraj-Palai/Voice2Vision/blob/master/run_on_colab.ipynb)**

### Steps:
1. **Enable GPU**: Click **Runtime** ➔ **Change runtime type** ➔ select **T4 GPU** ➔ **Save**.
2. **Step 1**: Click Play (▶) to install MOSS and AI libraries (~30 seconds).
3. **Step 2**: Click Play (▶) to load the models and launch the Gradio Web App.
4. Open the generated **public web URL** (`https://xxxx.gradio.live`) to use the application!

---

## 🧠 AI Models & Architecture

| Component | Technology / Model |
| :--- | :--- |
| **Speech-to-Text** | `OpenMOSS-Team/MOSS-Transcribe-Diarize` (0.9B CausalLM) |
| **Image Generation** | `dreamlike-art/dreamlike-diffusion-1.0` (Original Stable Diffusion, **No Watermark**) |
| **Video Generation** | `AnimateDiffPipeline` with `AnimateLCM` + `emilianJR/epiCRealism` (Real 16-Frame AI Video) |
| **Frontend UI** | Gradio Web App (`demo.launch(share=True)`) |

---

## 📁 Repository Structure

* **`run_on_colab.ipynb`**: Ready-to-run Colab notebook with the 2-step setup and Gradio web interface.
* **`voice to image and video model1.ipynb`**: The primary project notebook containing the complete authentic AI pipelines.
* **`app.py`**: Standalone backend server configured for GPU hosting or local run with NVIDIA CUDA.
* **`templates/` & `static/`**: Web frontend assets (HTML, CSS, JS).
