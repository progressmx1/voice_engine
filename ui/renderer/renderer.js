'use strict';

// Renderer for the Edith test UI. Talks to the main process only through the
// allow-listed `window.edith` API exposed by preload.js.

const $ = (id) => document.getElementById(id);

// ---- tabs -----------------------------------------------------------------

const tabs = ['stt', 'bench', 'tts'];
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    const name = btn.dataset.tab;
    tabs.forEach((t) => {
      $('tab-' + t).hidden = t !== name;
    });
  });
});

// ---- status ---------------------------------------------------------------

async function updateStatus() {
  try {
    const res = await window.edith.ping();
    $('status').textContent = `engine: ok (edith ${res.version})`;
    $('status').className = '';
  } catch (err) {
    $('status').textContent = `engine: ${err.message}`;
    $('status').className = 'error';
  }
}

// ---- STT ------------------------------------------------------------------

const state = {
  source: 'file',
  filePath: null,
  recording: null, // Float32Array of 16 kHz mono samples
  recorder: null,
  audioContext: null,
};

function setError(message) {
  const el = $('stt-error');
  el.textContent = message || '';
  el.hidden = !message;
}

function updateSttControls() {
  const isFile = state.source === 'file';
  const isMic = state.source === 'mic';
  $('stt-pick').disabled = !isFile;
  $('stt-record').disabled = !isMic || !!state.recorder;
  $('stt-stop').disabled = !state.recorder;
  $('stt-run').disabled = !(state.filePath || state.recording);
}

$('stt-source').addEventListener('change', (e) => {
  state.source = e.target.value;
  state.filePath = null;
  state.recording = null;
  $('stt-file-name').textContent = '';
  updateSttControls();
});

$('stt-pick').addEventListener('click', async () => {
  setError('');
  const path = await window.edith.pickAudioFile();
  if (!path) return;
  state.filePath = path;
  state.recording = null;
  $('stt-file-name').textContent = path;
  updateSttControls();
});

$('stt-record').addEventListener('click', async () => {
  setError('');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioContext({ sampleRate: 16000 });
    state.audioContext = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];

    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(ctx.destination);

    state.recorder = { stream, processor, chunks };
    state.recording = null;
    updateSttControls();
  } catch (err) {
    setError(`microphone: ${err.message}`);
  }
});

$('stt-stop').addEventListener('click', () => {
  const { stream, processor, chunks } = state.recorder;
  try {
    processor.disconnect();
    stream.getTracks().forEach((t) => t.stop());
  } catch {
    /* ignore */
  }
  if (state.audioContext) state.audioContext.close();
  state.audioContext = null;

  const length = chunks.reduce((n, c) => n + c.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  state.recording = samples;
  state.recorder = null;
  state.filePath = null;
  $('stt-file-name').textContent = `microphone recording (${(length / 16000).toFixed(1)} s)`;
  updateSttControls();
});

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, s) => {
    for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

$('stt-run').addEventListener('click', async () => {
  setError('');
  $('stt-text').textContent = 'transcribing…';
  $('stt-metrics').textContent = '…';
  const tbody = $('stt-segments').querySelector('tbody');
  tbody.innerHTML = '';

  try {
    let audioPath = state.filePath;
    if (state.recording) {
      audioPath = await window.edith.saveRecording(encodeWav(state.recording, 16000));
    }

    const result = await window.edith.transcribe({
      audio: audioPath,
      model: $('stt-model').value,
      device: $('stt-device').value,
      language: 'en',
    });

    $('stt-text').textContent = result.text || '(empty)';

    const latency = (result.rtf || 0) * (result.seconds || 0);
    $('stt-metrics').innerHTML = '';
    [
      ['RTF', (result.rtf || 0).toFixed(4)],
      ['Latency', latency.toFixed(3) + ' s'],
      ['Audio', (result.seconds || 0).toFixed(2) + ' s'],
      ['Model', result.model],
      ['Device', result.device],
      ['Compute', result.compute_type],
    ].forEach(([label, value]) => {
      const div = document.createElement('div');
      div.className = 'metric';
      const strong = document.createElement('strong');
      strong.textContent = value;
      const small = document.createElement('small');
      small.textContent = label;
      div.append(strong, small);
      $('stt-metrics').appendChild(div);
    });

    (result.segments || []).forEach((seg) => {
      const tr = document.createElement('tr');
      [seg.start.toFixed(2), seg.end.toFixed(2), seg.text].forEach((text) => {
        const td = document.createElement('td');
        td.textContent = text;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  } catch (err) {
    setError(err.message);
    $('stt-text').textContent = '—';
    $('stt-metrics').textContent = '—';
  }
});

// ---- TTS ------------------------------------------------------------------

const tts = {
  presets: [],
  clips: [], // { n, label, intonation, speed, result, url }
  current: null,
  busy: false,
};

function setTtsError(message) {
  const el = $('tts-error');
  el.textContent = message || '';
  el.hidden = !message;
}

function option(value, text) {
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = text;
  return opt;
}

function showSliderValues() {
  $('tts-mix-value').textContent = `${$('tts-mix').value}%`;
  $('tts-intonation-value').textContent = `${Number($('tts-intonation').value).toFixed(2)}×`;
  $('tts-speed-value').textContent = `${Number($('tts-speed').value).toFixed(2)}×`;
  const blending = $('tts-voice-b').value !== '';
  $('tts-mix').disabled = !blending;
  $('tts-voice-b').disabled = $('tts-voice-a').value === 'edith';
}

// The voice sent to the engine: a name, or a blend {a: 1 - mix, b: mix}.
function currentVoice() {
  const a = $('tts-voice-a').value;
  const b = $('tts-voice-b').value;
  if (!b || a === 'edith' || a === b) return a;
  const mix = Number($('tts-mix').value) / 100;
  return { [a]: Number((1 - mix).toFixed(2)), [b]: Number(mix.toFixed(2)) };
}

function voiceLabel(voice) {
  if (typeof voice === 'string') return voice;
  return Object.entries(voice)
    .map(([name, w]) => `${Math.round(w * 100)}% ${name}`)
    .join(' + ');
}

function applyPreset(preset) {
  const voice = preset.voice;
  if (typeof voice === 'string') {
    $('tts-voice-a').value = voice;
    $('tts-voice-b').value = '';
  } else {
    const [[a, wa], [b]] = Object.entries(voice);
    $('tts-voice-a').value = a;
    $('tts-voice-b').value = b;
    $('tts-mix').value = String(Math.round((1 - wa) * 100));
  }
  $('tts-intonation').value = String(preset.intonation);
  showSliderValues();
}

async function loadTtsVoices() {
  try {
    const res = await window.edith.ttsVoices({ device: $('tts-device').value });
    const a = $('tts-voice-a');
    const b = $('tts-voice-b');
    a.innerHTML = '';
    res.voices.forEach((v) => a.appendChild(option(v, v)));
    res.voices
      .filter((v) => v !== 'edith')
      .forEach((v) => b.appendChild(option(v, v)));
    tts.presets = res.presets;
    res.presets.forEach((p, i) => $('tts-preset').appendChild(option(String(i), p.name)));
    if (!$('tts-text').value) $('tts-text').value = res.default_text;
    $('tts-preset').value = '0';
    applyPreset(res.presets[0]);
    $('tts-speak').disabled = false;
  } catch (err) {
    setTtsError(`could not load voices: ${err.message}`);
  }
}

function metric(label, value) {
  const div = document.createElement('div');
  div.className = 'metric';
  const strong = document.createElement('strong');
  strong.textContent = value;
  const small = document.createElement('small');
  small.textContent = label;
  div.append(strong, small);
  return div;
}

function showClip(clip) {
  const r = clip.result;
  tts.current = clip;
  $('tts-audio').src = clip.url;
  $('tts-spoken').textContent = r.spoken || '—';
  $('tts-heard').textContent = clip.heard || '—';
  const m = r.measures;
  const items = [
    ['First audio', `${r.first_audio_ms.toFixed(0)} ms`],
    ['RTF', r.rtf.toFixed(3)],
    ['Length', `${r.seconds.toFixed(1)} s`],
    ['Device', r.device],
    ['Chunks', String(r.chunks.length)],
  ];
  if (m) {
    items.push(
      ['Pitch spread', `${m.f0_spread.toFixed(1)} st`],
      ['Median pitch', `${m.f0_median.toFixed(0)} Hz`],
      ['Jitter', `${(m.jitter * 100).toFixed(2)}%`],
      ['Shimmer', `${(m.shimmer * 100).toFixed(2)}%`],
    );
  }
  if (clip.wer != null) items.push(['Whisper WER', `${(clip.wer * 100).toFixed(1)}%`]);
  $('tts-metrics').innerHTML = '';
  items.forEach(([label, value]) => $('tts-metrics').appendChild(metric(label, value)));
  $('tts-check').disabled = tts.busy;
}

function addHistoryRow(clip) {
  const r = clip.result;
  const tr = document.createElement('tr');
  [
    String(clip.n),
    clip.label,
    `${clip.intonation.toFixed(2)}×`,
    `${clip.speed.toFixed(2)}×`,
    `${r.seconds.toFixed(1)} s`,
    r.measures ? `${r.measures.f0_spread.toFixed(1)} st` : '—',
    `${r.first_audio_ms.toFixed(0)} ms`,
  ].forEach((text) => {
    const td = document.createElement('td');
    td.textContent = text;
    tr.appendChild(td);
  });
  const td = document.createElement('td');
  const play = document.createElement('button');
  play.type = 'button';
  play.textContent = 'Play';
  play.addEventListener('click', () => {
    showClip(clip);
    $('tts-audio').play();
  });
  td.appendChild(play);
  tr.appendChild(td);
  $('tts-history').querySelector('tbody').prepend(tr);
}

function setTtsBusy(busy, label) {
  tts.busy = busy;
  $('tts-speak').disabled = busy;
  $('tts-check').disabled = busy || !tts.current;
  $('tts-speak').textContent = busy && label === 'speak' ? 'Speaking…' : 'Speak';
  $('tts-check').textContent = busy && label === 'check' ? 'Listening…' : 'Check with Whisper';
}

$('tts-speak').addEventListener('click', async () => {
  setTtsError('');
  const text = $('tts-text').value.trim();
  if (!text) {
    setTtsError('Type something for Edith to say.');
    return;
  }
  const voice = currentVoice();
  const intonation = Number($('tts-intonation').value);
  const speed = Number($('tts-speed').value);
  setTtsBusy(true, 'speak');
  try {
    const result = await window.edith.tts({
      text,
      voice,
      intonation,
      speed,
      device: $('tts-device').value,
    });
    const url = URL.createObjectURL(new Blob([result.wav], { type: 'audio/wav' }));
    const clip = {
      n: tts.clips.length + 1,
      label: voiceLabel(voice),
      intonation,
      speed,
      result,
      url,
      heard: null,
      wer: null,
    };
    tts.clips.push(clip);
    addHistoryRow(clip);
    showClip(clip);
    $('tts-audio').play();
  } catch (err) {
    setTtsError(err.message);
  } finally {
    setTtsBusy(false);
  }
});

$('tts-check').addEventListener('click', async () => {
  const clip = tts.current;
  if (!clip) return;
  setTtsError('');
  setTtsBusy(true, 'check');
  $('tts-heard').textContent = 'listening… (the first check loads Whisper, about a minute)';
  try {
    const res = await window.edith.ttsCheck({
      path: clip.result.path,
      spoken: clip.result.spoken,
      device: $('tts-device').value,
    });
    clip.heard = res.heard;
    clip.wer = res.wer;
    showClip(clip);
  } catch (err) {
    setTtsError(err.message);
    $('tts-heard').textContent = '—';
  } finally {
    setTtsBusy(false);
  }
});

$('tts-preset').addEventListener('change', (e) => {
  if (e.target.value !== '') applyPreset(tts.presets[Number(e.target.value)]);
});

['tts-voice-a', 'tts-voice-b', 'tts-mix', 'tts-intonation', 'tts-speed'].forEach((id) => {
  $(id).addEventListener('input', () => {
    $('tts-preset').value = '';
    showSliderValues();
  });
});

// ---- benchmark ------------------------------------------------------------

async function renderBench() {
  const el = $('bench-content');
  try {
    const doc = await window.edith.readBench();
    if (!doc || !Array.isArray(doc.results) || doc.results.length === 0) {
      el.textContent = 'No benchmark data in output/bench.json yet.';
      el.className = 'muted';
      return;
    }
    el.className = '';
    el.innerHTML = '';

    const generated = document.createElement('p');
    generated.className = 'muted';
    generated.textContent = `Generated ${doc.generated_at || 'unknown'}`;
    el.appendChild(generated);

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML =
      '<tr><th>Stage</th><th>Engine</th><th>Device</th><th>Audio (s)</th>' +
      '<th>Runs</th><th>RTF</th><th>WER</th><th>Latency (s)</th></tr>';
    const tbody = document.createElement('tbody');
    doc.results.forEach((r) => {
      const tr = document.createElement('tr');
      const wer = r.wer == null ? 'n/a' : r.wer.toFixed(4);
      [
        r.stage,
        r.engine,
        r.device,
        r.audio_seconds.toFixed(2),
        String(r.runs),
        r.rtf_mean.toFixed(4),
        wer,
        r.latency_mean.toFixed(4),
      ].forEach((text) => {
        const td = document.createElement('td');
        td.textContent = text;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    el.appendChild(table);

    if (doc.skipped && doc.skipped.length) {
      const skipped = document.createElement('p');
      skipped.className = 'muted';
      skipped.textContent = `Skipped: ${doc.skipped.join(', ')}`;
      el.appendChild(skipped);
    }
  } catch (err) {
    el.textContent = `Could not read output/bench.json: ${err.message}`;
    el.className = 'error';
  }
}

$('bench-refresh').addEventListener('click', renderBench);

// ---- init -----------------------------------------------------------------

updateStatus();
updateSttControls();
renderBench();
loadTtsVoices();
