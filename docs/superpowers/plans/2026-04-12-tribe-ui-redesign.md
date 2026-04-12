# TRIBE v2 UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current complex multi-panel UI with a clean, professional Command layout — full-viewport brain, 44px topbar, floating ROI strip, and a bottom input bar — while preserving all Three.js, WebSocket, and encoder logic.

**Architecture:** All changes are confined to `tribe-v2-playground.html`. The CSS `<style>` block is replaced wholesale with the new palette and layout classes. The HTML body is rebuilt to match the spec. JS is patched surgically — only the ~18 functions that touch DOM IDs or UI state need updating; Three.js, WebSocket, and encoder logic are untouched.

**Tech Stack:** Vanilla HTML/CSS/JS, Three.js (already embedded), Google Fonts (Inter + JetBrains Mono via CDN)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `tribe-v2-playground.html` lines 1–19 | `<head>` — add Google Fonts link |
| Replace | `tribe-v2-playground.html` lines 20–1254 | Entire `<style>` block |
| Replace | `tribe-v2-playground.html` lines 1256–1934 | Entire `<body>` HTML |
| Patch | `tribe-v2-playground.html` lines 1935–end | ~18 JS functions (surgical edits only) |

---

### Task 1: Replace CSS

**Files:**
- Modify: `tribe-v2-playground.html` (head + style block)

- [ ] **Step 1: Add Google Fonts to `<head>`**

Open `tribe-v2-playground.html`. Find the `<head>` block. Add the following line before the existing `<style>` tag (line ~19):

```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

- [ ] **Step 2: Replace the entire `<style>` block**

Delete everything from `<style>` through the closing `</style>` tag (~lines 20–1254) and replace with:

```css
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #0c0c0e;
  --surface: #141416;
  --overlay: #1a1a1d;
  --border:  rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.12);

  --t1: #f0f0f2;
  --t2: #888892;
  --t3: #44444e;

  --ac:    #5b7fff;
  --ac-bg: rgba(91,127,255,0.1);
  --red:   #e05252;

  --r-visual:     #e05252;
  --r-auditory:   #e07a42;
  --r-language:   #5b7fff;
  --r-prefrontal: #3db870;
  --r-motor:      #a855f7;
  --r-parietal:   #d4a429;

  --sans: 'Inter', system-ui, sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --r: 6px;
}

html, body {
  height: 100%;
  font-family: var(--sans);
  background: var(--bg);
  color: var(--t1);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* ── LAYOUT ─────────────────────────── */
.shell { display: flex; flex-direction: column; height: 100vh; }

/* ── TOPBAR ─────────────────────────── */
.topbar {
  height: 44px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 12px;
  flex-shrink: 0;
  background: var(--bg);
  z-index: 10;
}
.brand {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: .05em;
  color: var(--t1);
  white-space: nowrap;
}
.brand-v { color: var(--t3); margin-left: 4px; font-size: 11px; }

.topbar-divider { width: 1px; height: 18px; background: var(--border); flex-shrink: 0; }

.topbar-mid {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
}
.view-chip {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: .08em;
  color: var(--t3);
  padding: 3px 10px;
  border-radius: 20px;
  cursor: pointer;
  border: 1px solid transparent;
  user-select: none;
  transition: all .12s;
}
.view-chip:hover { color: var(--t2); border-color: var(--border); }
.view-chip.on { color: var(--t1); border-color: var(--border2); background: var(--overlay); }

.topbar-right { display: flex; align-items: center; gap: 6px; }
.icon-btn {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  border-radius: var(--r);
  border: 1px solid var(--border);
  color: var(--t3);
  cursor: pointer;
  font-size: 12px;
  user-select: none;
  transition: all .1s;
}
.icon-btn:hover { color: var(--t2); border-color: var(--border2); background: var(--overlay); }

/* ── MAIN ───────────────────────────── */
.main { flex: 1; position: relative; overflow: hidden; }

#brainCanvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

/* Loading overlay */
#loadingOverlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg);
  z-index: 5;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  letter-spacing: .06em;
}

/* ROI strip — left edge */
.roi-strip {
  position: absolute;
  left: 20px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  gap: 5px;
  z-index: 4;
}
.roi-row {
  display: flex;
  align-items: center;
  gap: 7px;
  cursor: pointer;
  opacity: 0.5;
  user-select: none;
  transition: opacity .12s;
}
.roi-row:hover { opacity: 0.85; }
.roi-row.on { opacity: 1; }
.roi-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.roi-name {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: .06em;
  color: var(--t2);
  text-transform: uppercase;
}

/* BOLD colorbar — right edge */
.cbar {
  position: absolute;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  z-index: 4;
}
.cbar-track {
  width: 6px;
  height: 100px;
  border-radius: 3px;
  background: linear-gradient(to bottom, #e05252, rgba(255,255,255,0.3), #3db870);
  opacity: 0.5;
}
.cbar-label {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--t3);
  letter-spacing: .04em;
}

/* Result card */
#resultCard {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--overlay);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 8px 16px;
  display: none;
  gap: 18px;
  white-space: nowrap;
  z-index: 4;
  align-items: center;
}
#resultCard.visible { display: flex; }
.rc-stat { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.rc-val {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 500;
  color: var(--t1);
}
.rc-key {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--t3);
  letter-spacing: .06em;
  text-transform: uppercase;
}

/* Info slide-in panel */
#infoPanel {
  position: absolute;
  top: 0; right: 0; bottom: 0;
  width: 280px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  z-index: 20;
  display: none;
  flex-direction: column;
  padding: 16px;
  gap: 12px;
  overflow-y: auto;
}
#infoPanel.open { display: flex; }
.info-title {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: .08em;
  color: var(--t3);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.enc-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.enc-name { font-family: var(--mono); font-size: 11px; color: var(--t2); }
.enc-status {
  font-family: var(--mono);
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 4px;
}
.enc-status.loaded { color: #3db870; }
.enc-status.missing { color: var(--t3); }
.dl-btn {
  width: 100%;
  padding: 6px 12px;
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--r);
  color: var(--t2);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: .06em;
  cursor: pointer;
  text-align: center;
  transition: all .1s;
  margin-top: 4px;
}
.dl-btn:hover { color: var(--t1); border-color: var(--border2); background: var(--overlay); }
.dl-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.dl-progress {
  height: 2px;
  background: var(--border);
  border-radius: 1px;
  margin-top: 4px;
  overflow: hidden;
  display: none;
}
.dl-progress.active { display: block; }
.dl-progress-fill {
  height: 100%;
  background: var(--ac);
  width: 0%;
  transition: width .2s;
}
.info-sep { height: 1px; background: var(--border); }

/* ── BOTTOM BAR ─────────────────────── */
.bottom-bar {
  border-top: 1px solid var(--border);
  padding: 10px 16px 12px;
  background: var(--bg);
  flex-shrink: 0;
  z-index: 10;
}
.mode-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}
.mode-tab {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: .10em;
  color: var(--t3);
  padding: 4px 14px 6px;
  cursor: pointer;
  border-bottom: 1.5px solid transparent;
  margin-bottom: -1px;
  user-select: none;
  transition: all .1s;
}
.mode-tab:hover { color: var(--t2); }
.mode-tab.on { color: var(--t1); border-bottom-color: var(--ac); }

/* File chip (shown when file attached) */
#fileChip {
  display: none;
  align-items: center;
  gap: 8px;
  padding: 5px 10px 5px 8px;
  background: var(--overlay);
  border: 1px solid var(--border2);
  border-radius: 6px;
  margin-bottom: 8px;
}
#fileChip.visible { display: flex; }
.fc-ext {
  font-family: var(--mono);
  font-size: 9px;
  background: var(--ac-bg);
  color: var(--ac);
  padding: 2px 5px;
  border-radius: 3px;
  letter-spacing: .04em;
  flex-shrink: 0;
}
.fc-name { font-family: var(--mono); font-size: 11px; color: var(--t2); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fc-size { font-family: var(--mono); font-size: 10px; color: var(--t3); flex-shrink: 0; }
.fc-rm { color: var(--t3); cursor: pointer; font-size: 12px; flex-shrink: 0; padding: 0 2px; }
.fc-rm:hover { color: var(--t1); }

.input-row {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  transition: border-color .1s;
}
.input-row.drag-over { border-color: var(--ac); border-style: dashed; }
.input-row:focus-within { border-color: var(--border2); }

#textInput {
  flex: 1;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--t2);
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  line-height: 1.5;
  min-height: 18px;
  max-height: 120px;
}
#textInput::placeholder { color: var(--t3); }

.dz-msg {
  flex: 1;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  pointer-events: none;
}

.seq-label {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--t3);
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
}
.seq-label:hover { color: var(--t2); }

.input-sep { width: 1px; height: 16px; background: var(--border); flex-shrink: 0; }

.run-btn {
  height: 28px;
  padding: 0 16px;
  background: var(--ac);
  border-radius: 5px;
  font-size: 11px;
  font-weight: 600;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
  letter-spacing: .01em;
  cursor: pointer;
  user-select: none;
  border: none;
  transition: opacity .1s;
}
.run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.run-btn.cancelling {
  background: transparent;
  border: 1px solid rgba(224,82,82,0.3);
  color: var(--red);
}
.kbd {
  font-size: 9px;
  opacity: 0.55;
  background: rgba(255,255,255,0.15);
  border-radius: 3px;
  padding: 1px 5px;
  font-family: var(--mono);
}

/* Infer progress track (inside bottom bar, below input-row) */
#inferTrack {
  height: 2px;
  background: var(--border);
  border-radius: 1px;
  margin-top: 8px;
  display: none;
  overflow: hidden;
}
#inferTrack.active { display: block; }
#inferFill {
  height: 100%;
  background: var(--ac);
  width: 0%;
  transition: width .3s;
}
</style>
```

- [ ] **Step 3: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "ui: replace CSS with new Command layout palette"
```

Expected: commit succeeds.

---

### Task 2: Replace HTML Body

**Files:**
- Modify: `tribe-v2-playground.html` (body HTML, lines ~1256–1934)

- [ ] **Step 1: Replace everything inside `<body>…</body>` (excluding the script tags)**

Find the `<body>` tag and delete all HTML down to (but not including) the first `<script>` tag. Replace with:

```html
<body>
<div class="shell">

  <!-- TOPBAR -->
  <div class="topbar">
    <div class="brand">TRIBE<span class="brand-v">v2</span></div>
    <div class="topbar-divider"></div>
    <div class="topbar-mid">
      <div class="view-chip on" id="vc-lateral" onclick="setCamView('lateral')">LATERAL</div>
      <div class="view-chip" id="vc-medial" onclick="setCamView('medial')">MEDIAL</div>
      <div class="view-chip" id="vc-dorsal" onclick="setCamView('dorsal')">DORSAL</div>
    </div>
    <div class="topbar-right">
      <div class="icon-btn" title="Export" onclick="exportResults()">&#x2193;</div>
      <div class="icon-btn" id="infoBtn" title="Info" onclick="toggleInfoPanel()">?</div>
    </div>
  </div>

  <!-- MAIN BRAIN AREA -->
  <div class="main" id="mainArea">
    <canvas id="brainCanvas"></canvas>

    <!-- Loading overlay -->
    <div id="loadingOverlay">LOADING BRAIN MESH…</div>

    <!-- ROI strip — left edge -->
    <div class="roi-strip">
      <div class="roi-row on" data-r="visual"     onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-visual);"></div>
        <div class="roi-name">Visual</div>
      </div>
      <div class="roi-row on" data-r="auditory"   onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-auditory);"></div>
        <div class="roi-name">Auditory</div>
      </div>
      <div class="roi-row on" data-r="language"   onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-language);"></div>
        <div class="roi-name">Language</div>
      </div>
      <div class="roi-row on" data-r="prefrontal" onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-prefrontal);"></div>
        <div class="roi-name">Prefrontal</div>
      </div>
      <div class="roi-row on" data-r="motor"      onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-motor);"></div>
        <div class="roi-name">Motor</div>
      </div>
      <div class="roi-row on" data-r="parietal"   onclick="toggleROI(this)">
        <div class="roi-dot" style="background:var(--r-parietal);"></div>
        <div class="roi-name">Parietal</div>
      </div>
    </div>

    <!-- BOLD colorbar — right edge -->
    <div class="cbar">
      <div class="cbar-label">+</div>
      <div class="cbar-track"></div>
      <div class="cbar-label">&minus;</div>
    </div>

    <!-- Result card (shown after prediction) -->
    <div id="resultCard"></div>

    <!-- Info slide-in panel -->
    <div id="infoPanel">
      <div class="info-title">Encoders</div>

      <!-- LLaMA -->
      <div>
        <div class="enc-row">
          <div class="enc-name">LLaMA (text)</div>
          <div class="enc-status missing" id="llamaStatus">not loaded</div>
        </div>
        <button class="dl-btn" id="dlLlama" onclick="startDownload('llama')">Download</button>
        <div class="dl-progress" id="dlLlamaProgress">
          <div class="dl-progress-fill" id="dlLlamaFill"></div>
        </div>
      </div>

      <div class="info-sep"></div>

      <!-- CLIP -->
      <div>
        <div class="enc-row">
          <div class="enc-name">CLIP (image)</div>
          <div class="enc-status missing" id="clipStatus">not loaded</div>
        </div>
        <button class="dl-btn" id="dlClip" onclick="startDownload('clip')">Download</button>
        <div class="dl-progress" id="dlClipProgress">
          <div class="dl-progress-fill" id="dlClipFill"></div>
        </div>
      </div>

      <div class="info-sep"></div>

      <!-- Wav2Vec -->
      <div>
        <div class="enc-row">
          <div class="enc-name">Wav2Vec (audio)</div>
          <div class="enc-status missing" id="wav2vecStatus">not loaded</div>
        </div>
        <button class="dl-btn" id="dlWav2vec" onclick="startDownload('wav2vec')">Download</button>
        <div class="dl-progress" id="dlWav2vecProgress">
          <div class="dl-progress-fill" id="dlWav2vecFill"></div>
        </div>
      </div>
    </div>

    <!-- Hidden compatibility stubs (preserve old JS IDs) -->
    <div id="statusBar"      style="display:none;"></div>
    <div id="resultPanel"    style="display:none;"></div>
    <div id="comparePanel"   style="display:none;"></div>
    <div id="sessionId"      style="display:none;"></div>
    <div id="demoMode"       style="display:none;"></div>
  </div>

  <!-- BOTTOM INPUT BAR -->
  <div class="bottom-bar">
    <!-- Mode tabs -->
    <div class="mode-tabs">
      <div class="mode-tab on" id="tab-text"  onclick="selStim('text')">TEXT</div>
      <div class="mode-tab"    id="tab-image" onclick="selStim('image')">IMAGE</div>
      <div class="mode-tab"    id="tab-audio" onclick="selStim('audio')">AUDIO</div>
    </div>

    <!-- File chip (image/audio mode) -->
    <div id="fileChip">
      <span class="fc-ext" id="fcExt"></span>
      <span class="fc-name" id="fcName"></span>
      <span class="fc-size" id="fcSize"></span>
      <span class="fc-rm" onclick="clearInput()">&#x2715;</span>
    </div>

    <!-- Input row -->
    <div class="input-row" id="inputRow">
      <!-- Text mode -->
      <textarea id="textInput" rows="1" placeholder="Describe a stimulus…" oninput="autoResize(this)"></textarea>
      <!-- Image drop zone msg (hidden when text active) -->
      <div class="dz-msg" id="dzImage" style="display:none;">Drop an image file here</div>
      <!-- Audio drop zone msg -->
      <div class="dz-msg" id="dzAudio" style="display:none;">Drop an audio file here</div>

      <!-- Seq control -->
      <div class="seq-label" id="seqLabel" onclick="cycleSeq()">seq 16</div>
      <div class="input-sep"></div>
      <!-- Run button -->
      <button class="run-btn" id="runBtn" onclick="runOrCancel()">Run <span class="kbd">&#x21B5;</span></button>
    </div>

    <!-- Infer progress track -->
    <div id="inferTrack"><div id="inferFill"></div></div>
  </div>

</div>
```

- [ ] **Step 2: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "ui: replace HTML body with Command layout structure"
```

---

### Task 3: Patch JS Functions

**Files:**
- Modify: `tribe-v2-playground.html` (JS section, all existing functions)

JS patches are surgical edits. For each, find the named function and replace only the stated lines.

- [ ] **Step 1: Update `init()` — wire canvas resize and hide loading overlay**

Find `function init()`. After `renderer.setSize(...)` (or wherever the Three.js renderer is sized), add a `ResizeObserver` on `#mainArea` and remove `#loadingOverlay` after mesh load:

```js
// After Three.js scene/renderer setup inside init():
const mainArea = document.getElementById('mainArea');
const resizeObs = new ResizeObserver(() => {
  const w = mainArea.clientWidth, h = mainArea.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});
resizeObs.observe(mainArea);

// After mesh is loaded/ready, call:
document.getElementById('loadingOverlay').style.display = 'none';
```

- [ ] **Step 2: Update `selStim(mode)` — switch tabs + show/hide drop zone messages**

Find the existing `selStim` function (or equivalent that switches TEXT/IMAGE/AUDIO). Replace its body with:

```js
function selStim(mode) {
  currentMode = mode;   // preserve existing state var name
  ['text','image','audio'].forEach(m => {
    document.getElementById('tab-' + m).classList.toggle('on', m === mode);
  });
  document.getElementById('textInput').style.display  = mode === 'text'  ? ''     : 'none';
  document.getElementById('dzImage').style.display    = mode === 'image' ? ''     : 'none';
  document.getElementById('dzAudio').style.display    = mode === 'audio' ? ''     : 'none';
  if (mode !== 'image' && mode !== 'audio') clearInput();
}
```

- [ ] **Step 3: Update `handleFile(file)` — show file chip**

Find the function that receives a dropped/selected file. After you have `file`, call `_showFileChip(file)`:

```js
function _showFileChip(file) {
  const ext = (file.name.split('.').pop() || '').toUpperCase();
  const kb  = (file.size / 1024).toFixed(0);
  const mb  = (file.size / 1048576).toFixed(1);
  const sz  = file.size > 1048576 ? mb + ' MB' : kb + ' KB';
  document.getElementById('fcExt').textContent  = ext;
  document.getElementById('fcName').textContent = file.name;
  document.getElementById('fcSize').textContent = sz;
  document.getElementById('fileChip').classList.add('visible');
}
```

Call `_showFileChip(file)` at the end of the existing file-handling block (after storing the file/base64).

- [ ] **Step 4: Update `clearInput()` — hide file chip**

Find `clearInput` (or equivalent reset function). Add at the start:

```js
document.getElementById('fileChip').classList.remove('visible');
document.getElementById('fcExt').textContent  = '';
document.getElementById('fcName').textContent = '';
document.getElementById('fcSize').textContent = '';
```

- [ ] **Step 5: Update `_setStatus(msg)` / status display**

Find wherever status messages are displayed (previously in a sidebar status area). Remove any references to old status bar IDs. If the function updates `document.getElementById('statusBar')`, that element still exists as a hidden stub — no crash, no change needed.

If there is a toast/overlay status mechanism, keep it unchanged. Result card is the primary output now.

- [ ] **Step 6: Update `cancelPrediction()` / run-button state**

Find the function that cancels inference. Replace button state logic:

```js
function _setRunning(running) {
  const btn = document.getElementById('runBtn');
  const track = document.getElementById('inferTrack');
  if (running) {
    btn.classList.add('cancelling');
    btn.innerHTML = 'Cancel';
    btn.disabled = false;
    track.classList.add('active');
    document.getElementById('inferFill').style.width = '0%';
  } else {
    btn.classList.remove('cancelling');
    btn.innerHTML = 'Run <span class="kbd">&#x21B5;</span>';
    btn.disabled = false;
    track.classList.remove('active');
  }
}
```

Call `_setRunning(true)` when inference starts, `_setRunning(false)` when it ends or is cancelled.

- [ ] **Step 7: Update `_handleWSMsg(msg)` — progress bar fill**

Find the WebSocket message handler. When a progress update arrives (e.g., `msg.type === 'progress'`), update the infer fill:

```js
if (msg.type === 'progress' && msg.pct != null) {
  document.getElementById('inferFill').style.width = msg.pct + '%';
}
```

- [ ] **Step 8: Update `setCamView(view)` — view chips**

Find `setCamView`. Replace the chip-highlight logic (previously updated sidebar buttons or menu items) with:

```js
['lateral','medial','dorsal'].forEach(v => {
  document.getElementById('vc-' + v).classList.toggle('on', v === view);
});
```

Keep the Three.js camera-repositioning code unchanged.

- [ ] **Step 9: Add `toggleInfoPanel()`**

Add a new function (near other UI helpers):

```js
function toggleInfoPanel() {
  const panel = document.getElementById('infoPanel');
  panel.classList.toggle('open');
}
```

- [ ] **Step 10: Update `fetchModelInfo()` — update info panel encoder status**

Find where encoder load status is checked (likely calls `GET /api/model_info` or similar). Update the DOM targets:

```js
// After receiving model info response, for each encoder:
// response shape assumed: { text_enc: bool, clip_enc: bool, audio_enc: bool }
function _applyModelInfo(info) {
  const setStatus = (id, loaded) => {
    const el = document.getElementById(id);
    el.textContent = loaded ? 'loaded' : 'not loaded';
    el.className = 'enc-status ' + (loaded ? 'loaded' : 'missing');
  };
  setStatus('llamaStatus',   info.text_enc);
  setStatus('clipStatus',    info.clip_enc);
  setStatus('wav2vecStatus', info.audio_enc);

  // Show/hide download buttons
  document.getElementById('dlLlama').style.display   = info.text_enc  ? 'none' : '';
  document.getElementById('dlClip').style.display    = info.clip_enc  ? 'none' : '';
  document.getElementById('dlWav2vec').style.display = info.audio_enc ? 'none' : '';
}
```

Call `_applyModelInfo(data)` wherever the existing fetch callback runs.

- [ ] **Step 11: Update `startDownload(model)` — wire info panel download buttons**

Find `startDownload` (Task 6 download logic). Update to use the new progress elements:

```js
function startDownload(model) {
  const progressId = { llama: 'dlLlamaProgress', clip: 'dlClipProgress', wav2vec: 'dlWav2vecProgress' }[model];
  const fillId     = { llama: 'dlLlamaFill',     clip: 'dlClipFill',     wav2vec: 'dlWav2vecFill'     }[model];
  const btnId      = { llama: 'dlLlama',          clip: 'dlClip',         wav2vec: 'dlWav2vec'          }[model];

  const btn  = document.getElementById(btnId);
  const prog = document.getElementById(progressId);
  const fill = document.getElementById(fillId);

  btn.disabled = true;
  prog.classList.add('active');
  fill.style.width = '0%';

  // existing fetch / websocket download logic — only update progress target:
  // wherever old code did barFill.style.width = pct + '%', replace with:
  //   fill.style.width = pct + '%';
  // wherever old code re-enabled button, replace with:
  //   btn.disabled = false; prog.classList.remove('active');
}
```

- [ ] **Step 12: Add `cycleSeq()`**

Add a new function:

```js
const SEQ_STEPS = [4, 8, 16, 32, 64];
function cycleSeq() {
  const label = document.getElementById('seqLabel');
  const cur = parseInt(label.textContent.replace('seq ',''), 10);
  const idx = SEQ_STEPS.indexOf(cur);
  const next = SEQ_STEPS[(idx + 1) % SEQ_STEPS.length];
  label.textContent = 'seq ' + next;
  seqLen = next;   // update the existing seqLen state variable
}
```

- [ ] **Step 13: Add `runOrCancel()` dispatcher**

Add:

```js
function runOrCancel() {
  const btn = document.getElementById('runBtn');
  if (btn.classList.contains('cancelling')) {
    cancelPrediction();
  } else {
    runPrediction();
  }
}
```

- [ ] **Step 14: Update `_applyResult(resp)` — populate result card**

Find the function that handles the prediction response and colors the brain. After the Three.js vertex-color update, add:

```js
function _showResultCard(resp) {
  // resp.temporal_acts: [[6 region scores per time step]]
  // Compute mean per region across time
  const regionNames  = ['Visual','Auditory','Language','Prefrontal','Motor','Parietal'];
  const regionColors = ['var(--r-visual)','var(--r-auditory)','var(--r-language)','var(--r-prefrontal)','var(--r-motor)','var(--r-parietal)'];
  const regionKeys   = ['visual','auditory','language','prefrontal','motor','parietal'];

  const T = resp.temporal_acts.length;
  const means = Array(6).fill(0);
  resp.temporal_acts.forEach(row => row.forEach((v, i) => { means[i] += v; }));
  means.forEach((_, i) => { means[i] /= T; });

  // Sort descending, take top 3
  const sorted = means
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3);

  const card = document.getElementById('resultCard');
  // Build stat nodes directly via DOM (no innerHTML)
  card.textContent = '';

  sorted.forEach(({ v, i }) => {
    const stat = document.createElement('div');
    stat.className = 'rc-stat';

    const val = document.createElement('div');
    val.className = 'rc-val';
    val.style.color = regionColors[i];
    val.textContent = v.toFixed(2);

    const key = document.createElement('div');
    key.className = 'rc-key';
    key.textContent = regionNames[i];

    stat.appendChild(val);
    stat.appendChild(key);
    card.appendChild(stat);
  });

  // Inference time
  if (resp.infer_ms != null) {
    const stat = document.createElement('div');
    stat.className = 'rc-stat';

    const val = document.createElement('div');
    val.className = 'rc-val';
    val.textContent = Math.round(resp.infer_ms).toString();

    const key = document.createElement('div');
    key.className = 'rc-key';
    key.textContent = 'ms';

    stat.appendChild(val);
    stat.appendChild(key);
    card.appendChild(stat);
  }

  card.classList.add('visible');
}
```

Call `_showResultCard(resp)` at the end of `_applyResult`.

- [ ] **Step 15: Add `autoResize(el)` for textarea**

```js
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
```

- [ ] **Step 16: Wire drag-and-drop on `#inputRow`**

Find where drop events are registered (likely on `document` or a drop zone). Add/replace with listeners on `#inputRow`:

```js
const inputRow = document.getElementById('inputRow');
inputRow.addEventListener('dragover', e => {
  e.preventDefault();
  inputRow.classList.add('drag-over');
});
inputRow.addEventListener('dragleave', () => inputRow.classList.remove('drag-over'));
inputRow.addEventListener('drop', e => {
  e.preventDefault();
  inputRow.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const isAudio = file.type.startsWith('audio/');
  selStim(isAudio ? 'audio' : 'image');
  handleFile(file);
});
```

- [ ] **Step 17: Wire keyboard shortcut on `#textInput`**

Find where `Enter` / `Cmd+Enter` triggers run. Update to use `#textInput`:

```js
document.getElementById('textInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    runOrCancel();
  }
});
```

- [ ] **Step 18: Remove old UI event listeners**

Find and delete any listeners that referenced the old sidebar panels, menu toggles, preset chips, comparison panel, or demo banner. Search for references to IDs like `sidePanel`, `menuBar`, `presetChips`, `comparePanel`, `demoBanner` and remove those listener registrations. The hidden stub divs (e.g., `#statusBar`, `#resultPanel`) will silently absorb any missed getElementById calls.

- [ ] **Step 19: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "ui: patch JS functions for Command layout"
```

---

### Task 4: Smoke Test

**Files:**
- Read: `tribe-v2-playground.html` (verify)
- Verify: browser (manual check)

- [ ] **Step 1: Serve the file**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
python3 -m http.server 8090
```

Open `http://localhost:8090/tribe-v2-playground.html` in a browser.

- [ ] **Step 2: Verify layout**

Check each of the following:
- [ ] Topbar is 44px, brand `TRIBE v2` in mono, `LATERAL/MEDIAL/DORSAL` chips center-aligned
- [ ] Brain canvas fills all space between topbar and bottom bar
- [ ] ROI strip floats on left edge, colorbar on right edge — no background panels
- [ ] Bottom bar has `TEXT / IMAGE / AUDIO` underline tabs and input row with `Run` button
- [ ] No old side panels, menus, preset chips, or gradients visible anywhere
- [ ] Color palette matches spec (`#0c0c0e` bg, `#5b7fff` accent, no gradients)

- [ ] **Step 3: Test mode switching**

- [ ] Click `IMAGE` tab — drop zone message appears, textarea hides
- [ ] Click `AUDIO` tab — audio drop zone message appears
- [ ] Click `TEXT` tab — textarea reappears
- [ ] Drop an image file onto input row — file chip appears with ext/name/size, mode switches to IMAGE
- [ ] Click `✕` on chip — chip disappears, returns to text mode

- [ ] **Step 4: Test camera views**

Click `MEDIAL` chip — brain rotates to medial view, chip highlights.
Click `DORSAL` chip — brain rotates to dorsal view.
Click `LATERAL` — returns to lateral view.

- [ ] **Step 5: Test `?` panel**

- [ ] Click `?` icon button — info slide-in panel appears on right
- [ ] Click again — panel closes
- [ ] Encoder statuses show (loaded/not loaded) based on server response

- [ ] **Step 6: Test inference (if server running)**

Start server:
```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
cargo run --release --bin tribe-server
```

Type a prompt, press Enter — result card appears at bottom of brain with top 3 region scores and inference time in ms.

- [ ] **Step 7: Commit verification**

```bash
git log --oneline -5
```

Expected: 3 new commits visible from Tasks 1–3.
