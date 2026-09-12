/* ══════════════════════════════════════════════════════════════
   VoiceGen AI – frontend logic
   ══════════════════════════════════════════════════════════════ */

"use strict";

// ── element refs ────────────────────────────────────────────────
const btnStart        = document.getElementById("btn-start");
const btnStop         = document.getElementById("btn-stop");
const btnTranscribe   = document.getElementById("btn-transcribe");
const btnGenImage     = document.getElementById("btn-gen-image");
const btnGenVideo     = document.getElementById("btn-gen-video");

const visualiserCanvas = document.getElementById("visualiser");
const visualiserCtx    = visualiserCanvas.getContext("2d");
const recTimer         = document.getElementById("rec-timer");
const visualiserWrap   = document.querySelector(".visualiser-wrap");

const playbackWrap    = document.getElementById("playback-wrap");
const playbackAudio   = document.getElementById("playback");

const fileUpload      = document.getElementById("file-upload");
const uploadFilename  = document.getElementById("upload-filename");

const transcriptStatus = document.getElementById("transcript-status");
const transcriptText   = document.getElementById("transcript-text");

const imageStatus     = document.getElementById("image-status");
const imageDisplay    = document.getElementById("image-display");
const downloadImage   = document.getElementById("download-image");

const videoStatus     = document.getElementById("video-status");
const videoDisplay    = document.getElementById("video-display");
const downloadVideo   = document.getElementById("download-video");

// ── state ────────────────────────────────────────────────────────
let mediaRecorder   = null;
let audioChunks     = [];
let audioBlob       = null;
let audioCtx        = null;
let analyser        = null;
let animFrameId     = null;
let timerInterval   = null;
let secondsElapsed  = 0;

// ══════════════════════════════════════════════════════════════
//  WAVEFORM VISUALISER
// ══════════════════════════════════════════════════════════════

function startVisualiser(stream) {
  audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  const source = audioCtx.createMediaStreamSource(stream);
  source.connect(analyser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray    = new Uint8Array(bufferLength);

  const W = visualiserCanvas.width;
  const H = visualiserCanvas.height;
  const barWidth = (W / bufferLength) * 2.5;

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    visualiserCtx.clearRect(0, 0, W, H);

    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * H;
      const hue = 240 + (i / bufferLength) * 60; // indigo → teal
      visualiserCtx.fillStyle = `hsl(${hue}, 80%, 60%)`;
      visualiserCtx.fillRect(x, H - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }
  draw();
}

function stopVisualiser() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (audioCtx)    audioCtx.close();
  visualiserCtx.clearRect(0, 0, visualiserCanvas.width, visualiserCanvas.height);
}

// ── timer ────────────────────────────────────────────────────────
function startTimer() {
  secondsElapsed = 0;
  recTimer.hidden = false;
  recTimer.textContent = "00:00";
  timerInterval = setInterval(() => {
    secondsElapsed++;
    const m = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
    const s = String(secondsElapsed % 60).padStart(2, "0");
    recTimer.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
}

// ══════════════════════════════════════════════════════════════
//  RECORDING
// ══════════════════════════════════════════════════════════════

btnStart.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioChunks  = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const url = URL.createObjectURL(audioBlob);
      playbackAudio.src = url;
      playbackWrap.hidden = false;
      stopVisualiser();
      stopTimer();
      recTimer.hidden = true;
      visualiserWrap.classList.remove("recording");
    };

    mediaRecorder.start(100);
    startVisualiser(stream);
    startTimer();
    visualiserWrap.classList.add("recording");

    btnStart.disabled = true;
    btnStop.disabled  = false;

    // reset downstream UI
    resetTranscript();
    resetResults();

  } catch (err) {
    alert("Microphone access denied or unavailable:\n" + err.message);
  }
});

btnStop.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  btnStart.disabled = false;
  btnStop.disabled  = true;
});

// ══════════════════════════════════════════════════════════════
//  FILE UPLOAD
// ══════════════════════════════════════════════════════════════

fileUpload.addEventListener("change", () => {
  const file = fileUpload.files[0];
  if (!file) return;
  uploadFilename.textContent = file.name;
  audioBlob = file;

  // show playback
  const url = URL.createObjectURL(file);
  playbackAudio.src = url;
  playbackWrap.hidden = false;

  resetTranscript();
  resetResults();
});

// ══════════════════════════════════════════════════════════════
//  TRANSCRIBE
// ══════════════════════════════════════════════════════════════

btnTranscribe.addEventListener("click", async () => {
  if (!audioBlob) {
    showStatus(transcriptStatus, "error", "No audio to transcribe.");
    return;
  }

  showStatus(transcriptStatus, "loading", "Transcribing… this may take a moment.");
  btnTranscribe.disabled = true;
  btnGenImage.disabled   = true;
  btnGenVideo.disabled   = true;
  transcriptText.value   = "";

  try {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");

    const res  = await fetch("/transcribe", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    transcriptText.value = data.transcript;
    showStatus(transcriptStatus, "success", "Transcription complete.");
    btnGenImage.disabled = false;
    btnGenVideo.disabled = false;

  } catch (err) {
    showStatus(transcriptStatus, "error", "Transcription failed: " + shortError(err));
  } finally {
    btnTranscribe.disabled = false;
  }
});

// allow manual edits to enable/disable generate buttons
transcriptText.addEventListener("input", () => {
  const hasText = transcriptText.value.trim().length > 0;
  btnGenImage.disabled = !hasText;
  btnGenVideo.disabled = !hasText;
});

// ══════════════════════════════════════════════════════════════
//  GENERATE IMAGE
// ══════════════════════════════════════════════════════════════

btnGenImage.addEventListener("click", async () => {
  const prompt = transcriptText.value.trim();
  if (!prompt) return;

  showStatus(imageStatus, "loading", "Generating image… please wait.");
  btnGenImage.disabled = true;
  imageDisplay.innerHTML = '<span class="placeholder-text">Generating…</span>';
  downloadImage.hidden = true;

  try {
    const res  = await fetch("/generate-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // bust cache so browser reloads the image
    const img = new Image();
    img.alt = "Generated image";
    img.src = data.image_url + "?t=" + Date.now();
    img.onload = () => {
      imageDisplay.innerHTML = "";
      imageDisplay.appendChild(img);
      downloadImage.href   = img.src;
      downloadImage.hidden = false;
    };
    showStatus(imageStatus, "success", "Image generated.");

  } catch (err) {
    showStatus(imageStatus, "error", "Image generation failed: " + shortError(err));
    imageDisplay.innerHTML = '<span class="placeholder-text">Generation failed.</span>';
  } finally {
    btnGenImage.disabled = false;
  }
});

// ══════════════════════════════════════════════════════════════
//  GENERATE VIDEO
// ══════════════════════════════════════════════════════════════

btnGenVideo.addEventListener("click", async () => {
  const prompt = transcriptText.value.trim();
  if (!prompt) return;

  showStatus(videoStatus, "loading", "Generating animated video… this takes ~30s.");
  btnGenVideo.disabled = true;
  videoDisplay.innerHTML = '<span class="placeholder-text">Generating…</span>';
  downloadVideo.hidden = true;

  try {
    const res  = await fetch("/generate-video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const img = new Image();
    img.alt = "Generated animation";
    img.src = data.video_url + "?t=" + Date.now();
    img.onload = () => {
      videoDisplay.innerHTML = "";
      videoDisplay.appendChild(img);
      downloadVideo.href   = img.src;
      downloadVideo.hidden = false;
    };
    showStatus(videoStatus, "success", "Video generated.");

  } catch (err) {
    showStatus(videoStatus, "error", "Video generation failed: " + shortError(err));
    videoDisplay.innerHTML = '<span class="placeholder-text">Generation failed.</span>';
  } finally {
    btnGenVideo.disabled = false;
  }
});

// ══════════════════════════════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════════════════════════════

/**
 * Show a status bar with a given type and message.
 * @param {HTMLElement} el   - the .status-bar element
 * @param {'loading'|'success'|'error'} type
 * @param {string} message
 */
function showStatus(el, type, message) {
  el.hidden = false;
  el.className = "status-bar " + type;
  const icon = type === "loading"
    ? '<span class="spinner"></span>'
    : type === "success" ? "✅" : "❌";
  el.innerHTML = `${icon} <span>${escapeHtml(message)}</span>`;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function shortError(err) {
  // Keep only the first line of a possibly long traceback
  return (err.message || String(err)).split("\n")[0].slice(0, 200);
}

function resetTranscript() {
  transcriptText.value  = "";
  transcriptStatus.hidden = true;
  btnGenImage.disabled  = true;
  btnGenVideo.disabled  = true;
}

function resetResults() {
  imageDisplay.innerHTML = '<span class="placeholder-text">Image will appear here</span>';
  videoDisplay.innerHTML = '<span class="placeholder-text">Video (GIF) will appear here</span>';
  imageStatus.hidden = true;
  videoStatus.hidden = true;
  downloadImage.hidden = true;
  downloadVideo.hidden = true;
}
