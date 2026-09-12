"""
Voice -> Image & Video Web Application
=======================================
Powered by the authentic AI models from 'voice to image and video model1.ipynb':
1. Transcription : OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B CausalLM)
2. Image Gen     : dreamlike-art/dreamlike-diffusion-1.0 (StableDiffusionPipeline)
3. Video Gen     : AnimateDiffPipeline (emilianJR/epiCRealism + AnimateLCM motion adapter & LoRA)

Endpoints:
---------
GET  /                   serve frontend UI (index.html)
POST /transcribe         multipart/form-data  { audio: <file> }
POST /generate-image     JSON  { "prompt": "..." }
POST /generate-video     JSON  { "prompt": "..." }
GET  /outputs/<filename> serve generated files
"""

import os
import sys
import time
import traceback
from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# ── Device & Precision Resolution ──────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
MOSS_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

print(f"[Voice2Vision] Runtime Device : {DEVICE}")
print(f"[Voice2Vision] CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Voice2Vision] GPU Name       : {torch.cuda.get_device_name(0)}")

# ── Lazy Global Model Handles ──────────────────────────────────────
_moss_model = None
_moss_processor = None
_image_pipe = None
_video_pipe = None


# ════════════════════════════════════════════════════════════════════
# 1. SPEECH-TO-TEXT: MOSS Transcribe-Diarize (from notebook Cell 0/2)
# ════════════════════════════════════════════════════════════════════

def _load_moss():
    global _moss_model, _moss_processor
    if _moss_model is None or _moss_processor is None:
        print("[Voice2Vision] Loading MOSS-Transcribe-Diarize model...")
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

        # Check if local clone exists in workspace or standard Colab location
        for candidate in [BASE_DIR / "MOSS-Transcribe-Diarize", Path("/content/MOSS-Transcribe-Diarize")]:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        _moss_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="sdpa" if DEVICE.type == "cuda" else None,
        )
        _moss_model = _moss_model.to(dtype=MOSS_DTYPE).to(DEVICE)
        _moss_model.eval()

        print("[Voice2Vision] Loading MOSS processor...")
        _moss_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        print("[Voice2Vision] MOSS loaded successfully.")

    return _moss_model, _moss_processor


def _transcribe_audio_with_moss(audio_path: str) -> str:
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    model, processor = _load_moss()
    messages = build_transcription_messages(audio_path)

    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=2048,
        do_sample=False,
        device=DEVICE,
        dtype=MOSS_DTYPE,
    )

    segments = parse_transcript(result["text"])
    transcript = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )
    return transcript


# ════════════════════════════════════════════════════════════════════
# 2. IMAGE GENERATION: Dreamlike Diffusion 1.0 (from notebook Cell 0/3)
# ════════════════════════════════════════════════════════════════════

def _load_image_pipe():
    global _image_pipe
    if _image_pipe is None:
        print("[Voice2Vision] Loading Dreamlike Diffusion 1.0 Pipeline...")
        from diffusers import StableDiffusionPipeline

        model_id = "dreamlike-art/dreamlike-diffusion-1.0"
        _image_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        if DEVICE.type == "cuda":
            _image_pipe = _image_pipe.to(DEVICE)
        else:
            _image_pipe = _image_pipe.to("cpu")

        print("[Voice2Vision] Image Pipeline loaded successfully.")
    return _image_pipe


# ════════════════════════════════════════════════════════════════════
# 3. VIDEO GENERATION: AnimateDiff + AnimateLCM (from notebook Cell 0/3)
# ════════════════════════════════════════════════════════════════════

def _load_video_pipe():
    global _video_pipe
    if _video_pipe is None:
        print("[Voice2Vision] Loading AnimateDiff + AnimateLCM Pipeline...")
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter

        # Load Motion Adapter
        adapter = MotionAdapter.from_pretrained(
            "wangfuyun/AnimateLCM",
            torch_dtype=DTYPE,
        )

        # Load Base Model
        _video_pipe = AnimateDiffPipeline.from_pretrained(
            "emilianJR/epiCRealism",
            motion_adapter=adapter,
            torch_dtype=DTYPE,
        )

        # LCM Scheduler
        _video_pipe.scheduler = LCMScheduler.from_config(
            _video_pipe.scheduler.config,
            beta_schedule="linear",
        )

        # Load AnimateLCM LoRA
        _video_pipe.load_lora_weights(
            "wangfuyun/AnimateLCM",
            weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
            adapter_name="lcm-lora",
        )
        _video_pipe.set_adapters(["lcm-lora"], [0.8])

        if DEVICE.type == "cuda":
            _video_pipe.enable_model_cpu_offload()
        else:
            _video_pipe = _video_pipe.to("cpu")

        print("[Voice2Vision] Video Pipeline loaded successfully.")
    return _video_pipe


NEGATIVE_PROMPT = (
    "bad pose, awkward pose, unnatural posture, broken posture, "
    "broken fingers, malformed fingers, extra fingers, missing fingers, "
    "deformed hands, deformed feet, deformed legs, deformed arms, "
    "long limbs, short limbs, twisted limbs, unnatural joints, "
    "facial asymmetry, distorted mouth, distorted teeth, distorted nose, "
    "distorted ears, deformed eyes, uneven eyes, blinking artifacts, "
    "face melting, face warping, face morphing, identity drift, "
    "skin flickering, hair flickering, hair changing, "
    "clothing deformation, clothing flickering, clothing morphing, "
    "accessory deformation, jewelry deformation"
)


# ════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)


# ── POST /transcribe ─────────────────────────────────────────────────
@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"transcript": "", "error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        ext = Path(audio_file.filename or "recording.wav").suffix or ".wav"
        audio_path = str(OUTPUTS_DIR / f"input_audio{ext}")
        audio_file.save(audio_path)

        # Transcribe with MOSS-Transcribe-Diarize from notebook
        transcript = _transcribe_audio_with_moss(audio_path)

        return jsonify({"transcript": transcript, "error": None})

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"transcript": "", "error": err}), 500


# ── POST /generate-image ─────────────────────────────────────────────
@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"image_url": "", "error": "Prompt is empty"}), 400

        pipe = _load_image_pipe()
        generator = torch.Generator(device="cpu").manual_seed(int(time.time()) % 100000)
        image = pipe(prompt, generator=generator).images[0]

        out_path = OUTPUTS_DIR / "output_image.png"
        image.save(str(out_path))

        return jsonify({
            "image_url": f"/outputs/output_image.png?t={int(time.time())}",
            "error": None
        })

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"image_url": "", "error": err}), 500


# ── POST /generate-video ─────────────────────────────────────────────
@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"video_url": "", "error": "Prompt is empty"}), 400

        from diffusers.utils import export_to_gif

        pipe = _load_video_pipe()
        generator = torch.Generator(device="cpu").manual_seed(0)

        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=16,
            guidance_scale=2.0,
            num_inference_steps=6,
            generator=generator,
        )
        frames = output.frames[0]

        out_path = str(OUTPUTS_DIR / "output_video.gif")
        export_to_gif(frames, out_path)

        return jsonify({
            "video_url": f"/outputs/output_video.gif?t={int(time.time())}",
            "error": None
        })

    except Exception:
        err = traceback.format_exc()
        app.logger.error(err)
        return jsonify({"video_url": "", "error": err}), 500


# ── GET /status ──────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "status": "online",
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {
            "transcription": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "image": "dreamlike-art/dreamlike-diffusion-1.0",
            "video": "AnimateDiff (epiCRealism + AnimateLCM)"
        }
    })


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
