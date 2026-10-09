// Voice questions for the chat: record in the browser, encode 16 kHz mono WAV,
// and hand it to Dash through the `st-voice` store. A Python callback then
// sends it to the STT endpoint. Needs a secure context (https or localhost).
(function () {
  "use strict";

  const TARGET_RATE = 16000;
  const MAX_SECONDS = 60;
  let recorder = null;
  let stream = null;
  let stopTimer = null;

  // Update React-owned nodes only through Dash, never by touching the DOM.
  function setProps(id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(id, props);
    }
  }

  function showState(recording) {
    setProps("chat-mic", {
      children: recording ? "Stop" : "Record",
      className: recording ? "btn btn--ghost recording" : "btn btn--ghost",
    });
    setProps("chat-voice-status", {
      children: recording ? "Recording… press Stop to send your question." : "",
    });
  }

  function writeText(view, offset, text) {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  }

  function encodeWav(samples) {
    const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
    writeText(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeText(view, 8, "WAVE");
    writeText(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, TARGET_RATE, true);
    view.setUint32(28, TARGET_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    return new Uint8Array(view.buffer);
  }

  async function toWav(blob) {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * TARGET_RATE)), TARGET_RATE);
      const source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      return encodeWav((await offline.startRendering()).getChannelData(0));
    } finally {
      context.close();
    }
  }

  function toBase64(bytes) {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
  }

  async function finish(chunks, type) {
    try {
      const wav = await toWav(new Blob(chunks, { type }));
      setProps("st-voice", { data: { wav: toBase64(wav), t: Date.now() } });
    } catch (error) {
      setProps("chat-voice-status", { children: "The recording could not be processed. Try again." });
    }
  }

  function stop() {
    window.clearTimeout(stopTimer);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  async function start() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setProps("chat-voice-status", { children: "Voice recording is not supported in this browser." });
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      setProps("chat-voice-status", { children: "Microphone access was denied." });
      return;
    }
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.addEventListener("dataavailable", (event) => event.data.size && chunks.push(event.data));
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      showState(false);
      setProps("chat-voice-status", { children: "Transcribing question…" });
      finish(chunks, recorder.mimeType);
      recorder = null;
    });
    recorder.start();
    showState(true);
    stopTimer = window.setTimeout(stop, MAX_SECONDS * 1000);
  }

  document.addEventListener("click", (event) => {
    if (!event.target.closest("#chat-mic")) return;
    if (recorder) stop();
    else start();
  });
})();
