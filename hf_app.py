"""
Voice2Vision – Hugging Face Spaces entry point
===============================================
Transcription : OpenAI Whisper  (local CPU)
Image         : Pollinations.ai free API
Video         : Pollinations.ai frames → animated GIF (CPU)

Run locally : python hf_app.py
HF Spaces   : set "app_file: hf_app.py" in README.md front-matter
"""

import io
import time
import traceback
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image
import gradio as gr

# ── output folder ────────────────────────────────────────────────
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── lazy Whisper ─────────────────────────────────────────────────
_whisper = None

def _load_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model("base")
    return _whisper


# ════════════════════════════════════════════════════════════════
#  AUDIO LOADING  (no ffmpeg required)
# ════════════════════════════════════════════════════════════════

def _load_audio(path: str) -> np.ndarray:
    SR = 16_000

    # 1. soundfile  (WAV / FLAC / OGG natively)
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        return data.astype(np.float32)
    except Exception:
        pass

    # 2. librosa
    try:
        import librosa
        data, _ = librosa.load(path, sr=SR, mono=True)
        return data.astype(np.float32)
    except Exception:
        pass

    raise RuntimeError("Cannot decode audio – install soundfile or librosa.")


# ════════════════════════════════════════════════════════════════
#  POLLINATIONS.AI  (free, no API key)
# ════════════════════════════════════════════════════════════════

def _poll_image(prompt: str, width=768, height=512, seed=42) -> Image.Image:
    encoded = urllib.parse.quote(prompt)
    url = (f"https://image.pollinations.ai/prompt/{encoded}"
           f"?width={width}&height={height}&seed={seed}&nologo=true")
    req = urllib.request.Request(url, headers={"User-Agent": "Voice2Vision/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def _make_gif(prompt: str, n_frames=6, duration=180) -> str:
    frames = []
    for i in range(n_frames):
        img = _poll_image(prompt, width=512, height=512, seed=1000 + i * 17)
        # slight zoom-pan per frame for motion feel
        scale  = 1.0 + i * 0.015
        nw, nh = int(img.width * scale), int(img.height * scale)
        big    = img.resize((nw, nh), Image.LANCZOS)
        lx     = (nw - img.width)  // 2
        ly     = (nh - img.height) // 2
        frames.append(big.crop((lx, ly, lx + img.width, ly + img.height)))

    gif_path = str(OUT / "output_video.gif")
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        loop=0, duration=duration, optimize=False,
    )
    return gif_path


# ════════════════════════════════════════════════════════════════
#  GRADIO PIPELINE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path):
    """Whisper transcription – called by Gradio with a filepath."""
    if audio_path is None:
        return "", "⚠️ No audio recorded."
    try:
        audio_arr = _load_audio(audio_path)
        model     = _load_whisper()
        result    = model.transcribe(audio_arr, fp16=False)
        text      = result["text"].strip()
        return text, "✅ Transcription complete."
    except Exception:
        return "", f"❌ {traceback.format_exc()}"


def generate_image(prompt):
    """Fetch image from Pollinations.ai."""
    if not prompt.strip():
        return None, "⚠️ Prompt is empty."
    try:
        img      = _poll_image(prompt)
        out_path = str(OUT / "output_image.png")
        img.save(out_path)
        return out_path, "✅ Image generated."
    except Exception:
        return None, f"❌ {traceback.format_exc()}"


def generate_video(prompt):
    """Build animated GIF from Pollinations.ai frames."""
    if not prompt.strip():
        return None, "⚠️ Prompt is empty."
    try:
        gif_path = _make_gif(prompt)
        return gif_path, "✅ Video generated."
    except Exception:
        return None, f"❌ {traceback.format_exc()}"


# ════════════════════════════════════════════════════════════════
#  GRADIO UI
# ════════════════════════════════════════════════════════════════

CSS = """
body { background: #0d0f1a; color: #e2e4f0; font-family: 'Segoe UI', sans-serif; }
.gr-button-primary { background: linear-gradient(135deg,#7c6af7,#5eead4) !important; border:none !important; }
.gr-button { border-radius: 8px !important; font-weight: 600 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="🎙️ Voice2Vision AI") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:20px 0 10px'>
      <h1 style='font-size:2rem;margin:0'>
        🎙️ Voice<span style='background:linear-gradient(90deg,#7c6af7,#5eead4);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent'>2</span>Vision
      </h1>
      <p style='color:#7a7f9a;margin:6px 0 0'>Speak a scene — watch it come alive</p>
    </div>
    """)

    # ── Step 1 : Record ──────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 🎤 Step 1 — Record Your Voice")
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload audio (WAV / MP3 / OGG)",
        )
        transcribe_btn = gr.Button("📝 Transcribe", variant="primary")

    # ── Step 2 : Transcript ──────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 📝 Step 2 — Transcript / Prompt")
        transcript_box = gr.Textbox(
            label="Transcribed text (edit if needed)",
            placeholder="Your transcribed text will appear here…",
            lines=3,
        )
        trans_status = gr.Textbox(label="Status", interactive=False, lines=1)

    # ── Step 3 : Generate ────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 🖼️ Step 3 — Generate")
        with gr.Row():
            img_btn = gr.Button("🖼️ Generate Image", variant="primary")
            vid_btn = gr.Button("🎞️ Generate Video", variant="primary")

    # ── Step 4 : Results ─────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### ✨ Results")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**🖼️ Generated Image**")
                image_out  = gr.Image(label="Image", type="filepath")
                image_status = gr.Textbox(label="", interactive=False, lines=1)
            with gr.Column():
                gr.Markdown("**🎞️ Generated Video (GIF)**")
                video_out  = gr.Image(label="Video", type="filepath")
                video_status = gr.Textbox(label="", interactive=False, lines=1)

    gr.HTML("<p style='text-align:center;color:#7a7f9a;font-size:.8rem;margin-top:20px'>"
            "Powered by Whisper · Pollinations.ai · Gradio</p>")

    # ── wire up ──────────────────────────────────────────────────
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[transcript_box, trans_status],
    )
    img_btn.click(
        fn=generate_image,
        inputs=transcript_box,
        outputs=[image_out, image_status],
    )
    vid_btn.click(
        fn=generate_video,
        inputs=transcript_box,
        outputs=[video_out, video_status],
    )

# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch()
