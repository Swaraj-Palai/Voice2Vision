"""Run this to see the exact errors from each component."""
import traceback
from pathlib import Path

BASE = Path(__file__).parent
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── TEST 1: Whisper ──────────────────────────────────────────────
print("="*60)
print("TEST 1: Whisper transcription")
print("="*60)
try:
    import whisper
    model = whisper.load_model("base")
    # find any audio file in outputs to test with
    audio_files = list(OUTPUTS.glob("input_audio*"))
    if audio_files:
        result = model.transcribe(str(audio_files[0]), fp16=False)
        print("SUCCESS:", result["text"])
    else:
        print("No input audio found in outputs/ — skipping transcribe test")
        print("Whisper import OK, model loads OK")
except Exception:
    print("FAILED:")
    traceback.print_exc()

# ── TEST 2: Stable Diffusion ─────────────────────────────────────
print()
print("="*60)
print("TEST 2: Stable Diffusion image generation")
print("="*60)
try:
    import torch
    from diffusers import StableDiffusionPipeline
    print("diffusers import OK")
    print("CUDA available:", torch.cuda.is_available())
    pipe = StableDiffusionPipeline.from_pretrained(
        "dreamlike-art/dreamlike-diffusion-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    img = pipe("a red apple on a table", num_inference_steps=5).images[0]
    img.save(str(OUTPUTS / "test_image.png"))
    print("SUCCESS: test_image.png saved")
except Exception:
    print("FAILED:")
    traceback.print_exc()

# ── TEST 3: AnimateDiff ──────────────────────────────────────────
print()
print("="*60)
print("TEST 3: AnimateDiff video generation")
print("="*60)
try:
    import torch
    from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
    from diffusers.utils import export_to_gif
    print("diffusers AnimateDiff import OK")
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe.load_lora_weights(
        "wangfuyun/AnimateLCM",
        weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
        adapter_name="lcm-lora",
    )
    pipe.set_adapters(["lcm-lora"], [0.8])
    pipe.enable_model_cpu_offload()
    output = pipe(
        prompt="a red apple on a table",
        num_frames=4,
        guidance_scale=2.0,
        num_inference_steps=4,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    export_to_gif(output.frames[0], str(OUTPUTS / "test_video.gif"))
    print("SUCCESS: test_video.gif saved")
except Exception:
    print("FAILED:")
    traceback.print_exc()
