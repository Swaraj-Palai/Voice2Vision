"""
Voice → Image & Video  (CPU-friendly, no GPU required)
=======================================================
Transcription : OpenAI Whisper  (local, CPU)
Image         : Pollinations.ai free API  (cloud, no key needed)
Video         : Pollinations.ai images stitched into an animated GIF (CPU)

Endpoints
---------
POST /transcribe       multipart/form-data  { audio: <file> }
POST /generate-image   JSON  { "prompt": "..." }
POST /generate-video   JSON  { "prompt": "..." }
GET  /outputs/<file>   serve generated files
"""

import io
import time
import traceback
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── paths ────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB max upload

# ── lazy Whisper handle ──────────────────────────────────────────
_whisper_model = None

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")   # ~140 MB, CPU-fine
    return _whisper_model


# ════════════════════════════════════════════════════════════════
#  AUDIO LOADING  (no ffmpeg required)
# ════════════════════════════════════════════════════════════════

def _load_audio(path: str) -> np.ndarray:
    """
    Return a float32 mono 16 kHz numpy array.
    Tries soundfile → librosa → pydub in order.
    """
    SR = 16_000

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

    try:
        import librosa
        data, _ = librosa.load(path, sr=SR, mono=True)
        return data.astype(np.float32)
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        seg = (AudioSegment.from_file(path)
               .set_channels(1)
               .set_frame_rate(SR))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0
        return samples
    except Exception:
        pass

    raise RuntimeError(
        "Cannot decode audio. Install soundfile, librosa, or pydub."
    )


# ════════════════════════════════════════════════════════════════
#  POLLINATIONS.AI  — free image API, no key needed
# ════════════════════════════════════════════════════════════════

def _fetch_image_from_pollinations(prompt: str,
                                   width: int = 768,
                                   height: int = 512,
                                   seed: int | None = None) -> Image.Image:
    """
    GET https://image.pollinations.ai/prompt/<encoded-prompt>?...
    Returns a PIL Image.
    """
    encoded = urllib.parse.quote(prompt)
    params  = f"width={width}&height={height}&nologo=true"
    if seed is not None:
        params += f"&seed={seed}"
    url = f"https://image.pollinations.ai/prompt/{encoded}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "VoiceGenAI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    return Image.open(io.BytesIO(data)).convert("RGB")


def _make_animated_gif(prompt: str,
                       n_frames: int = 6,
                       duration_ms: int = 200) -> str:
    """
    Fetch n_frames slightly-varied images from Pollinations and
    stitch them into an animated GIF.  Pure CPU, no torch needed.
    """
    frames = []
    for i in range(n_frames):
        img = _fetch_image_from_pollinations(
            prompt,
            width=512, height=512,
            seed=1000 + i * 17          # different seed → slightly different frame
        )
        # small zoom/shift per frame to give motion illusion
        scale = 1.0 + i * 0.015
        new_w = int(img.width  * scale)
        new_h = int(img.height * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        # centre-crop back to original size
        left = (new_w - img.width)  // 2
        top  = (new_h - img.height) // 2
        frame = resized.crop((left, top, left + img.width, top + img.height))
        frames.append(frame)

    out_path = str(OUTPUTS_DIR / "output_video.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    return out_path


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)


# ── POST /transcribe ─────────────────────────────────────────────
@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"transcript": "", "error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        ext        = Path(audio_file.filename or "rec.wav").suffix or ".wav"
        audio_path = str(OUTPUTS_DIR / f"input_audio{ext}")
        audio_file.save(audio_path)

        audio_array = _load_audio(audio_path)
        model       = _load_whisper()
        result      = model.transcribe(audio_array, fp16=False)
        transcript  = result["text"].strip()

        return jsonify({"transcript": transcript, "error": None})

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"transcript": "", "error": err}), 500


# ── POST /generate-image ─────────────────────────────────────────
@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        data   = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"image_url": "", "error": "Prompt is empty"}), 400

        img      = _fetch_image_from_pollinations(prompt, width=768, height=512)
        out_path = OUTPUTS_DIR / "output_image.png"
        img.save(str(out_path))

        return jsonify({"image_url": f"/outputs/output_image.png?t={int(time.time())}",
                        "error": None})

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"image_url": "", "error": err}), 500


# ── POST /generate-video ─────────────────────────────────────────
@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        data   = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"video_url": "", "error": "Prompt is empty"}), 400

        _make_animated_gif(prompt, n_frames=6, duration_ms=180)

        return jsonify({"video_url": f"/outputs/output_video.gif?t={int(time.time())}",
                        "error": None})

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"video_url": "", "error": err}), 500


# ── GET /debug ────────────────────────────────────────────────────
@app.route("/debug")
def debug():
    import subprocess, sys
    checks = {
        "flask":      "import flask; print(flask.__version__)",
        "whisper":    "import whisper; print(whisper.__version__)",
        "soundfile":  "import soundfile; print(soundfile.__version__)",
        "PIL":        "from PIL import Image; print(Image.__version__)",
        "numpy":      "import numpy; print(numpy.__version__)",
        "requests":   "import urllib.request; print('ok')",
    }
    lines = []
    for pkg, code in checks.items():
        try:
            out = subprocess.check_output(
                [sys.executable, "-c", code],
                stderr=subprocess.STDOUT, text=True
            ).strip()
            lines.append(f"✅ {pkg}: {out}")
        except subprocess.CalledProcessError as e:
            lines.append(f"❌ {pkg}: {e.output.strip()}")
    return "<pre style='font:14px monospace;padding:20px'>" + "\n".join(lines) + "</pre>"


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
