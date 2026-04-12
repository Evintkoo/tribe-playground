# TRIBE v2 — Fix & Live Brain Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all broken server/client wiring and upgrade the visualization into a live, per-vertex brain activity showcase supporting text, image, and audio inputs.

**Architecture:** The FastAPI inference server becomes the single origin — serving the HTML, brain mesh, WebSocket progress stream, and REST API. Full 20,484-vertex activations are returned and mapped onto the brain mesh for genuine per-vertex BOLD coloring instead of region-mean+noise. CLIP ViT-L/14 is added as a Python image encoder routed through the video projector slot (otherwise unused). Temporal per-region data enables TR-by-TR animation.

**Tech Stack:** FastAPI, PyTorch, transformers (CLIP, Wav2Vec2-BERT, LLaMA), safetensors, Pillow, Three.js, native WebSocket

---

## Bugs being fixed

| Bug | Root cause |
|-----|-----------|
| No WebSocket `/ws` route | Server only has REST endpoints |
| HTTP fallback hits `/api/predict` but server serves `/predict` | Path mismatch |
| HTML and `brain.obj` unserved by inference server | No static file mounting |
| `image_b64` silently ignored | No ImageEncoder in Python backend |
| `vertex_sample` ignored by frontend; only regional noise used | Frontend comment says "unused" |
| Video silently submits zeros | V-JEPA2 not integrated, no user feedback |

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `inference_server.py` | Modify | Serve static files; add `/ws` WS; fix route prefixes; add ImageEncoder; return full vertex data + temporal data |
| `tribe-v2-playground.html` | Modify | Fix vertex_acts usage; genuine per-vertex coloring with rank array; temporal animation; video overlay; encoder status dot; toast helper |

---

## Task 1: Serve static files and fix API route prefixes

**Files:**
- Modify: `inference_server.py`

- [ ] **Step 1: Add FileResponse import and static routes**

Add to the imports block after `from fastapi import FastAPI, HTTPException`:

```python
from fastapi.responses import FileResponse
```

Add these two routes immediately after the `app.add_middleware(...)` call (around line 387):

```python
@app.get("/")
def serve_html():
    p = os.path.join(SCRIPT_DIR, "tribe-v2-playground.html")
    if not os.path.exists(p):
        raise HTTPException(404, "tribe-v2-playground.html not found")
    return FileResponse(p, media_type="text/html")

@app.get("/brain.obj")
def serve_brain_obj():
    p = os.path.join(SCRIPT_DIR, "brain.obj")
    if not os.path.exists(p):
        raise HTTPException(404, "brain.obj not found")
    return FileResponse(p, media_type="application/octet-stream")
```

- [ ] **Step 2: Add `/api/` prefix to all three existing route decorators**

```python
# was: @app.post("/predict", ...)
@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest): ...

# was: @app.get("/health")
@app.get("/api/health")
def health(): ...

# was: @app.get("/info")
@app.get("/api/info")
def model_info(): ...
```

- [ ] **Step 3: Verify server starts and serves correctly**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
pkill -f inference_server.py 2>/dev/null; sleep 1
python3 inference_server.py &
sleep 5
curl -s http://127.0.0.1:8081/ | head -3
curl -s http://127.0.0.1:8081/api/health
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/brain.obj
```

Expected:
```
<!DOCTYPE html>
<html lang="en">
<head>
{"ready":true}
200
```

- [ ] **Step 4: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add inference_server.py
git commit -m "fix: serve HTML and brain.obj, prefix all API routes with /api/"
```

---

## Task 2: WebSocket `/ws` + refactor predict into `_predict_core`

**Files:**
- Modify: `inference_server.py`

The frontend connects to `ws://host/ws` at page load. No WebSocket route exists, so `_wsReady` stays false forever and the HTTP fallback path (`/api/predict`) was also broken (now fixed in Task 1). We add the WS endpoint and refactor the predict body into a shared sync function.

- [ ] **Step 1: Add WebSocket + asyncio imports**

Extend the import from fastapi on the same line (or add separately):

```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
import json as _json
```

- [ ] **Step 2: Update PredictRequest and PredictResponse models**

Replace the existing models (around lines 392-407) with:

```python
class PredictRequest(BaseModel):
    text:       str = ""
    audio_b64:  str = ""
    image_b64:  str = ""   # base64 image bytes (JPEG / PNG / WebP)
    seq_len:    int = 16
    subject_id: int = 0


class PredictResponse(BaseModel):
    region_stats:  dict
    global_stats:  dict
    vertex_acts:   list   # 20484 floats (mean over T) for per-vertex BOLD coloring
    temporal_acts: list   # [T][6] regional means per timepoint for TR animation
    seq_len:       int
    modality:      str
    elapsed_ms:    float
    demo_mode:     bool
```

- [ ] **Step 3: Rename the predict body to `_predict_core` and keep HTTP endpoint thin**

Rename the existing `predict()` function to `_predict_core(req: PredictRequest) -> PredictResponse` and update:

1. The guard check at the top — add `image_b64` to the empty-input check:
   ```python
   if not req.text.strip() and not req.audio_b64 and not req.image_b64:
       raise HTTPException(400, "Provide text, audio, or image input")
   ```

2. Add image encoding after the audio block (image routes through the video projector slot):
   ```python
   image_feat = None
   if req.image_b64 and not req.audio_b64 and not req.text.strip():
       modalities.append("image")
       if IMAGE_ENCODER is not None:
           try:
               image_bytes = base64.b64decode(req.image_b64)
               image_feat  = IMAGE_ENCODER.encode(image_bytes, seq)
           except Exception as e:
               print(f"[tribe] Image encoding error: {e}", flush=True)
               demo_mode = True
       else:
           demo_mode = True
           text_feat = text_to_features_demo("", seq)
   ```

3. Pass `image_feat` as the video argument:
   ```python
   with torch.no_grad():
       out = MODEL(text=text_feat, audio=audio_feat, video=image_feat)
   ```

4. Replace the 512-sample return with full vertex + temporal data:
   ```python
   act      = out[0].numpy()          # [T, 20484]
   rstats, gstats = region_stats(act)

   mean_act    = act.mean(axis=0)     # [20484]
   vertex_acts = [round(float(v), 4) for v in mean_act]

   region_order = ["visual","auditory","language","prefrontal","motor","parietal"]
   temporal_acts: list = []
   for t_idx in range(act.shape[0]):
       row = []
       for r in region_order:
           lo, hi = REGION_RANGES[r]
           row.append(round(float(act[t_idx, lo:hi].mean()), 4))
       temporal_acts.append(row)

   elapsed = (time.perf_counter() - t0) * 1000
   return PredictResponse(
       region_stats  = rstats,
       global_stats  = gstats,
       vertex_acts   = vertex_acts,
       temporal_acts = temporal_acts,
       seq_len       = seq,
       modality      = "+".join(modalities) if modalities else "unknown",
       elapsed_ms    = round(elapsed, 1),
       demo_mode     = demo_mode,
   )
   ```

5. Replace the HTTP endpoint body:
   ```python
   @app.post("/api/predict", response_model=PredictResponse)
   def predict(req: PredictRequest):
       return _predict_core(req)
   ```

- [ ] **Step 4: Add WebSocket endpoint**

Add after the HTTP endpoints (before the `if __name__` block):

```python
@app.websocket("/ws")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    async def send(msg: dict):
        await websocket.send_text(_json.dumps(msg))

    try:
        while True:
            raw = await websocket.receive_text()
            req_data = _json.loads(raw)

            await send({"type": "progress", "pct": 5,  "msg": "Received request"})

            try:
                req = PredictRequest(
                    text       = req_data.get("text", ""),
                    audio_b64  = req_data.get("audio_b64", ""),
                    image_b64  = req_data.get("image_b64", ""),
                    seq_len    = int(req_data.get("seq_len", 16)),
                    subject_id = int(req_data.get("subject_id", 0)),
                )
                await send({"type": "progress", "pct": 20, "msg": "Encoding stimulus"})
                result = await loop.run_in_executor(None, lambda: _predict_core(req))
                await send({"type": "progress", "pct": 90, "msg": "Coloring brain"})
                payload = result.model_dump()
                payload["type"] = "result"
                await send(payload)
            except HTTPException as e:
                await send({"type": "error", "message": e.detail})
            except Exception as e:
                await send({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
```

- [ ] **Step 5: Test WebSocket end-to-end**

```bash
pip install websockets --quiet
python3 -c "
import asyncio, websockets, json

async def run():
    async with websockets.connect('ws://127.0.0.1:8081/ws') as ws:
        await ws.send(json.dumps({'text': 'red sunset', 'seq_len': 4}))
        for _ in range(6):
            msg = json.loads(await ws.recv())
            t = msg.get('type')
            print(t, msg.get('pct',''), msg.get('msg', msg.get('modality','')))
            if t in ('result','error'):
                if t == 'result':
                    print('vertex_acts len:', len(msg.get('vertex_acts', [])))
                break

asyncio.run(run())
"
```

Expected:
```
progress 5 Received request
progress 20 Encoding stimulus
progress 90 Coloring brain
result  text
vertex_acts len: 20484
```

- [ ] **Step 6: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add inference_server.py
git commit -m "feat: WebSocket /ws with progress streaming; _predict_core refactor; full vertex + temporal data"
```

---

## Task 3: CLIP ViT-L/14 ImageEncoder

**Files:**
- Modify: `inference_server.py`

CLIP ViT-L/14 has 24 transformer layers, hidden size 1024. After `tribe_group_mean` on layers 12/18/24 we get `[n_patches, 2048]`. Zero-padding to 2816 fits the video projector.

- [ ] **Step 1: Add ImageEncoder class after TextEncoder**

```python
# ── Image encoder (CLIP ViT-L/14) ────────────────────────────────────────────

class ImageEncoder:
    """
    Encodes images using CLIP ViT-L/14 vision transformer.
    Extracts patch hidden states at layers 50%, 75%, 100% (12, 18, 24).
    tribe_group_mean -> [n_patches, 2048].
    Temporal pool to seq_len then zero-pad to 2816 (video projector slot).
    Returns [1, seq_len, 2816].
    """
    def __init__(self, weights_path: str):
        from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
        from safetensors.torch import load_file

        print("[tribe] Loading ImageEncoder (CLIP ViT-L/14)…", flush=True)
        cfg = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPVisionModel(cfg)
        self.model.eval()

        sd_full   = load_file(weights_path, device="cpu")
        prefix    = "vision_model."
        sd_vision = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
        if sd_vision:
            self.model.vision_model.load_state_dict(sd_vision, strict=False)
        else:
            # weights file may have no prefix (pure vision model save)
            self.model.load_state_dict(sd_full, strict=False)

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        n_layers = cfg.num_hidden_layers   # 24
        self.li  = [round(0.50 * n_layers), round(0.75 * n_layers), n_layers]
        print(f"[tribe] ImageEncoder ready ✓  (layers {self.li})", flush=True)

    def encode(self, image_bytes: bytes, seq_len: int) -> torch.Tensor:
        from PIL import Image
        import io
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # hidden_states: [1, n_patches+1, 1024]; index 0 = embed layer, 1..24 = transformer
        # Drop CLS token (position 0), keep patch tokens
        h      = [out.hidden_states[i].squeeze(0)[1:].float() for i in self.li]
        feats  = tribe_group_mean(*h)                           # [n_patches, 2048]
        pooled = temporal_pool(feats, seq_len)                  # [seq_len, 2048]
        pad    = torch.zeros(seq_len, 768, dtype=pooled.dtype)
        return torch.cat([pooled, pad], dim=-1).unsqueeze(0)    # [1, seq_len, 2816]
```

- [ ] **Step 2: Add global declaration and load in lifespan**

Near line 355, with MODEL/AUDIO_ENCODER/TEXT_ENCODER:
```python
IMAGE_ENCODER: Optional[ImageEncoder] = None
```

In `_lifespan`, after the TextEncoder block:
```python
clip_path = os.path.join(SCRIPT_DIR, "tribe-v2-weights", "clip", "model.safetensors")
try:
    if os.path.exists(clip_path):
        IMAGE_ENCODER = ImageEncoder(clip_path)
    else:
        print(f"[tribe] CLIP weights not found — run: python3 download_clip.py", flush=True)
except Exception as e:
    print(f"[tribe] ImageEncoder not loaded: {e}", flush=True)
    IMAGE_ENCODER = None
```

- [ ] **Step 3: Update /api/info encoders dict**

```python
"encoders": {
    "text":  ("LLaMA-3.2-3B · loaded"      if TEXT_ENCODER  else "demo · hash-based fallback"),
    "audio": ("Wav2Vec-BERT 2.0 · loaded"   if AUDIO_ENCODER else "not loaded"),
    "image": ("CLIP ViT-L/14 · loaded"      if IMAGE_ENCODER else "not loaded"),
    "video": "not loaded (V-JEPA2 not integrated)",
},
```

- [ ] **Step 4: Install Pillow**

```bash
pip install Pillow --quiet
python3 -c "from PIL import Image; print('Pillow OK')"
```

Expected: `Pillow OK`

- [ ] **Step 5: Test image encoding via HTTP**

```bash
python3 - << 'EOF'
import base64, requests, urllib.request, os
# Download a tiny public-domain image for testing
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
dest = "/tmp/_tribe_test.png"
if not os.path.exists(dest):
    urllib.request.urlretrieve(url, dest)
with open(dest, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post("http://127.0.0.1:8081/api/predict",
                  json={"image_b64": b64, "seq_len": 4}, timeout=60)
d = r.json()
print("modality:", d.get("modality"))
print("vertex_acts len:", len(d.get("vertex_acts", [])))
print("temporal frames:", len(d.get("temporal_acts", [])))
print("demo_mode:", d.get("demo_mode"))
EOF
```

Expected (CLIP loaded):
```
modality: image
vertex_acts len: 20484
temporal frames: 4
demo_mode: False
```

Expected (CLIP not downloaded yet):
```
modality: image
vertex_acts len: 20484
temporal frames: 4
demo_mode: True
```

- [ ] **Step 6: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add inference_server.py
git commit -m "feat: CLIP ViT-L/14 ImageEncoder; image_b64 support; expose image status in /api/info"
```

---

## Task 4: Frontend — genuine per-vertex BOLD coloring

**Files:**
- Modify: `tribe-v2-playground.html`

`setBrainVertexSample` currently ignores its arguments (the comment at line ~2997 says "unused"). We precompute a rank array in `onGeoReady` so every mesh vertex maps O(1) to a model vertex index, then apply `boldColormap` to the real model output.

- [ ] **Step 1: Add region ranges constant in the module script**

In the `<script type="module">` block, right after the `REGION_HEX` constant (around line 2670), add:

```javascript
// Mirrors Python REGION_RANGES in inference_server.py
const REGION_RANGES_JS = {
  visual:     [0,     3600],
  auditory:   [3600,  6800],
  language:   [6800,  10500],
  prefrontal: [10500, 14000],
  motor:      [14000, 17200],
  parietal:   [17200, 20484],
};
```

- [ ] **Step 2: Precompute vertRankInRegion in onGeoReady**

In `onGeoReady`, after the line `ALL_R.forEach(r=>{const n=regionVerts[r].length;if(n>0)regionCentroids[r].divideScalar(n);});` (around line 2835), add:

```javascript
// Precompute each vertex's ordinal rank within its region for O(1) model-vertex lookup
const _rankArr    = new Int32Array(N);
const _regCntMap  = {};
ALL_R.forEach(r => { _regCntMap[r] = 0; });
for (let vi = 0; vi < N; vi++) {
  const r = vertRegion[vi];
  _rankArr[vi] = _regCntMap[r]++;
}
window._vertRankInRegion = _rankArr;
window._regionVertCount  = _regCntMap;
```

- [ ] **Step 3: Replace setBrainVertexSample with genuine per-vertex mapping**

Find the existing `window.setBrainVertexSample=function(...)` (around line 2999) and replace the entire function:

```javascript
window.setBrainVertexSample = function(vertexActs, gMin, gMax, activeRegs, regionStats) {
  if (!brainReady || !vertRegion) return;
  const N         = colTgt.length / 3;
  const range     = Math.max(Math.abs(gMin), Math.abs(gMax), 0.001);
  const activeSet = new Set(activeRegs);
  const MODEL_N   = 20484;
  const rankArr   = window._vertRankInRegion;
  const regCntMap = window._regionVertCount;

  for (let i = 0; i < N; i++) {
    const r = vertRegion[i];
    if (!activeSet.has(r)) {
      colTgt[i*3]   = BASE_COL.r * 0.36;
      colTgt[i*3+1] = BASE_COL.g * 0.36;
      colTgt[i*3+2] = BASE_COL.b * 0.36;
      continue;
    }
    // Map mesh vertex rank -> model vertex index proportionally within region
    const lo  = REGION_RANGES_JS[r][0];
    const hi  = REGION_RANGES_JS[r][1];
    const cnt = regCntMap ? (regCntMap[r] || 1) : 1;
    const modelIdx = lo + Math.round((rankArr[i] / cnt) * (hi - lo));
    const act = vertexActs[Math.min(modelIdx, MODEL_N - 1)] ?? 0;
    const t   = Math.max(-1, Math.min(1, act / range));
    const c   = boldColormap(t);
    colTgt[i*3]   = c.r;
    colTgt[i*3+1] = c.g;
    colTgt[i*3+2] = c.b;
  }
};
```

- [ ] **Step 4: Update _applyResult to use vertex_acts**

In `_applyResult` (around line 2371), replace the `setBrainVertexSample` call block:

```javascript
if (window.setBrainVertexSample && (data.vertex_acts?.length > 0 || data.vertex_sample?.length > 0)) {
  const verts = data.vertex_acts?.length > 0 ? data.vertex_acts : data.vertex_sample;
  window.setBrainVertexSample(
    verts,
    data.global_stats.global_min,
    data.global_stats.global_max,
    ST.regions,
    data.region_stats
  );
}
```

- [ ] **Step 5: Increase LERP speed for snappier updates**

Find `const LERP=0.04;` (around line 3072) and change to:

```javascript
const LERP = 0.10;
```

- [ ] **Step 6: Verify in browser**

Type a sentence at `http://127.0.0.1:8081`. After prediction, zoom into a single region — you should see heterogeneous activation (varying shades) within the region rather than a uniform block, because real model vertex values now drive the per-vertex color.

- [ ] **Step 7: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "feat: genuine per-vertex BOLD coloring from 20484 model vertices"
```

---

## Task 5: Temporal TR animation

**Files:**
- Modify: `tribe-v2-playground.html`

`temporal_acts` from the server is `[T][6]` — T timepoints x 6 regions. We add a Play/Pause button and a scrubber in the viz toolbar that animate region-level activation labels through all T frames.

- [ ] **Step 1: Add temporal state variables and functions in main script**

In the main `<script>` block after the `_pinnedA`/`_diffActive` declarations (around line 2396), add:

```javascript
let _temporalActs    = null;   // [T][6] from server
let _temporalFrame   = 0;
let _temporalPlaying = false;
let _temporalTimer   = null;

function _applyTemporalFrame(t) {
  if (!_temporalActs || t >= _temporalActs.length) return;
  _temporalFrame = t;
  const regionOrder = ['visual','auditory','language','prefrontal','motor','parietal'];
  const frameStats  = {};
  regionOrder.forEach((r, i) => {
    const v = _temporalActs[t][i] ?? 0;
    frameStats[r] = { rel_activation: v, mean: v, std: 0.12, peak: Math.abs(v) };
  });
  if (window.updateLabelActivations) updateLabelActivations(frameStats);
  if (window.setBreathing) window.setBreathing(frameStats, ST.regions);
  if (window.setRegionGlow) window.setRegionGlow(frameStats, ST.regions);
  const scrubber = document.getElementById('temporalScrubber');
  if (scrubber) scrubber.value = t;
  const lbl = document.getElementById('temporalFrameLbl');
  if (lbl) lbl.textContent = 'TR ' + (t + 1) + ' / ' + _temporalActs.length;
}

function playTemporal() {
  if (!_temporalActs) return;
  _temporalPlaying = !_temporalPlaying;
  const btn = document.getElementById('vt-temporal');
  if (btn) { btn.classList.toggle('on', _temporalPlaying); btn.textContent = _temporalPlaying ? '\u23F8 TR' : '\u25B6 TR'; }
  if (_temporalPlaying) {
    _temporalTimer = setInterval(() => {
      _applyTemporalFrame((_temporalFrame + 1) % _temporalActs.length);
    }, 200);
  } else {
    clearInterval(_temporalTimer);
    if (_lastResult) {
      if (window.updateLabelActivations) updateLabelActivations(_lastResult.region_stats);
      if (window.setBreathing) window.setBreathing(_lastResult.region_stats, ST.regions);
      if (window.setRegionGlow) window.setRegionGlow(_lastResult.region_stats, ST.regions);
    }
  }
}
```

- [ ] **Step 2: Wire temporal data in _applyResult**

In `_applyResult`, after the comparison panel update (around line 2392), add:

```javascript
if (data.temporal_acts?.length > 0) {
  clearInterval(_temporalTimer);
  _temporalPlaying = false;
  _temporalFrame   = 0;
  _temporalActs    = data.temporal_acts;
  const btn = document.getElementById('vt-temporal');
  if (btn) { btn.style.display = ''; btn.textContent = '\u25B6 TR'; btn.classList.remove('on'); }
  const scrubber = document.getElementById('temporalScrubber');
  if (scrubber) { scrubber.max = data.temporal_acts.length - 1; scrubber.value = 0; }
  const lbl = document.getElementById('temporalFrameLbl');
  if (lbl) lbl.textContent = 'TR 1 / ' + data.temporal_acts.length;
  const ctrl = document.getElementById('temporalControls');
  if (ctrl) ctrl.style.display = 'flex';
}
```

- [ ] **Step 3: Add temporal controls to viz toolbar HTML**

In the viz toolbar `<div class="viz-toolbar">`, find the `<span class="vt-coord">` element (around line 1565) and insert before it:

```html
      <div class="vt-sep"></div>
      <button class="vt-btn" id="vt-temporal" onclick="playTemporal()" style="display:none" title="Animate temporal sequence">&#9654; TR</button>
      <div id="temporalControls" style="display:none;align-items:center;gap:5px">
        <input type="range" id="temporalScrubber" min="0" max="15" value="0"
               style="width:55px;height:2px;cursor:pointer;accent-color:var(--ac)"
               oninput="clearInterval(_temporalTimer);_temporalPlaying=false;document.getElementById('vt-temporal').classList.remove('on');document.getElementById('vt-temporal').textContent='\u25B6 TR';_applyTemporalFrame(+this.value)">
        <span id="temporalFrameLbl" style="font-size:9px;color:var(--t3);font-family:var(--mono);white-space:nowrap">TR 1</span>
      </div>
```

- [ ] **Step 4: Test in browser**

1. Run prediction on any text or image
2. A `▶ TR` button appears in the toolbar
3. Click it: region activation labels animate at ~5fps through all T timepoints
4. Drag the scrubber: jumps to any specific TR
5. Click `⏸ TR`: stops and restores mean activation

- [ ] **Step 5: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "feat: temporal TR animation with play/pause button and scrubber"
```

---

## Task 6: Video unavailable overlay + encoder status dot + toast

**Files:**
- Modify: `tribe-v2-playground.html`

- [ ] **Step 1: Add CSS for overlay and toast**

In the `<style>` block, before `</style>`, add:

```css
/* ── VIDEO UNAVAILABLE OVERLAY ────────────────────────── */
.unavail-overlay {
  position: absolute; inset: 0; border-radius: var(--r8);
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 8px; padding: 24px; text-align: center;
  background: rgba(5,7,12,0.88); backdrop-filter: blur(6px);
  z-index: 10; pointer-events: none;
}
.unavail-icon  { color: var(--orange); margin-bottom: 4px; }
.unavail-title { font-size: 12px; font-weight: 600; color: var(--t2); }
.unavail-sub   { font-size: 11px; color: var(--t3); max-width: 210px; line-height: 1.5; }

/* ── TOAST ───────────────────────────────────────────── */
#_toast {
  position: fixed; bottom: 72px; left: 50%; transform: translateX(-50%);
  background: rgba(20,24,40,0.96); color: var(--t2); font-size: 12px;
  padding: 8px 14px; border-radius: 6px; border: 1px solid var(--b1);
  z-index: 9999; pointer-events: none;
  transition: opacity 0.3s; opacity: 0; white-space: nowrap;
}
```

- [ ] **Step 2: Add overlay inside video panel and make panel position:relative**

Find `<div class="input-panel" id="panel-video">` (around line 1711). Add `style="position:relative"` to that div, and insert this as the first child:

```html
      <div class="unavail-overlay" id="videoUnavail">
        <div class="unavail-icon">
          <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.5" width="36" height="36" stroke-linecap="round">
            <rect x="2" y="6" width="20" height="20" rx="3"/>
            <path d="M22 12L30 8V24L22 20Z"/>
            <line x1="3" y1="3" x2="29" y2="29" stroke="var(--orange)" stroke-width="2"/>
          </svg>
        </div>
        <div class="unavail-title">Video encoder not yet integrated</div>
        <div class="unavail-sub">V-JEPA2 ViT-G is planned but not loaded. Use text, image, or audio instead.</div>
      </div>
```

- [ ] **Step 3: Add showToast function and block video in runPrediction**

In the main `<script>` block after `clearCompare()`, add:

```javascript
function showToast(msg, ms) {
  let t = document.getElementById('_toast');
  if (!t) { t = document.createElement('div'); t.id = '_toast'; document.body.appendChild(t); }
  t.textContent = msg;
  t.style.opacity = '1';
  clearTimeout(t._tid);
  t._tid = setTimeout(() => { t.style.opacity = '0'; }, ms || 3000);
}
```

In `runPrediction()`, insert before `_predRunning = true` (the payload assembly block):

```javascript
if (ST.stim === 'video') {
  showToast('Video encoder not available — use text, image, or audio.');
  return;
}
```

- [ ] **Step 4: Add encoder status dot to image panel meta**

Find in the image panel meta bar:
```html
<span class="meta-item">encoder: <span id="imageEncoderLabel">CLIP ViT-L/14</span></span>
```

Replace with:
```html
<span class="meta-item">encoder: <span id="imageEncoderLabel">CLIP ViT-L/14</span><span id="imgEncDot" title="checking…" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--t3);margin-left:5px;vertical-align:middle"></span></span>
```

In `fetchModelInfo()`, after the `imgLbl` update:

```javascript
const imgDot = document.getElementById('imgEncDot');
if (imgDot && d.encoders?.image) {
  const loaded = d.encoders.image.includes('loaded');
  imgDot.style.background = loaded ? 'var(--green)' : 'var(--orange)';
  imgDot.title = d.encoders.image;
}
```

- [ ] **Step 5: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "feat: video unavailable overlay, image encoder status dot, toast notifications"
```

---

## Task 7: Colorbar range display + activation-scaled glow

**Files:**
- Modify: `tribe-v2-playground.html`

- [ ] **Step 1: Add range line to colorbar HTML**

Find the colorbar div (around line 1614):
```html
<div class="colorbar" id="bColorbar">
  <div class="cb-title">BOLD Activation</div>
  <div class="cb-scale"></div>
  <div class="cb-labels"><span>Suppressed</span><span>0</span><span>Active</span></div>
</div>
```

Replace with:
```html
<div class="colorbar" id="bColorbar">
  <div class="cb-title">BOLD Activation</div>
  <div class="cb-scale"></div>
  <div class="cb-labels"><span>Suppressed</span><span>0</span><span>Active</span></div>
  <div id="cbRangeLine" style="font-size:9px;color:var(--t3);margin-top:3px;font-family:var(--mono);text-align:center"></div>
</div>
```

In `_applyResult`, after `cb.classList.add('visible')`:

```javascript
const cbRange = document.getElementById('cbRangeLine');
if (cbRange && data.global_stats) {
  const gs = data.global_stats;
  cbRange.textContent = '[' + gs.global_min.toFixed(3) + ', ' + gs.global_max.toFixed(3) + ']  20,484v';
}
```

- [ ] **Step 2: Scale glow sphere by activation level**

In the animate loop `ALL_R.forEach` that handles glow (around line 3106), replace the `glowMeshes[r].scale.setScalar(...)` line with:

```javascript
const lv = activationLevels[r] ?? 1;
const baseScale = 1.0 + Math.max(0, lv - 0.5) * 0.5;
glowMeshes[r].scale.setScalar(baseScale + 0.06 * Math.sin(t * 1.9 + idx * 1.2));
```

And the light intensity line:
```javascript
const tgtInt = on ? (gl.intensityTarget * (0.4 + lv * 0.6) + 0.15 * Math.sin(t * 2.3 + idx * .8)) : 0;
```

- [ ] **Step 3: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "feat: colorbar shows [min, max] + vertex count; glow sphere scales with activation level"
```

---

## Task 8: End-to-end verification

- [ ] **Step 1: Full restart and smoke-test all routes**

```bash
pkill -f inference_server.py 2>/dev/null; sleep 1
cd /Users/evintleovonzko/Documents/research/tribe-playground
python3 inference_server.py &
sleep 6
curl -s http://127.0.0.1:8081/api/health
curl -s http://127.0.0.1:8081/ | head -3
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/brain.obj
```

Expected:
```
{"ready":true}
<!DOCTYPE html>
<html lang="en">
200
```

- [ ] **Step 2: Verify text prediction returns new response shape**

```bash
curl -s -X POST http://127.0.0.1:8081/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"a dog running through a forest","seq_len":8}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('modality:', d['modality']); print('vertex_acts:', len(d.get('vertex_acts',[]))); print('temporal:', len(d.get('temporal_acts',[])), 'x', len(d.get('temporal_acts',[[]])[0]))"
```

Expected:
```
modality: text
vertex_acts: 20484
temporal: 8 x 6
```

- [ ] **Step 3: Browser checklist**

Open `http://127.0.0.1:8081` and verify:

| Check | Expected |
|-------|----------|
| Page loads | Brain renders, no 404s in DevTools network tab |
| `brain.obj` | Actual brain mesh (not procedural fallback) |
| WebSocket | No "WebSocket is closed" errors in console |
| Type text → 1.5s | Progress bar animates 5% → 20% → 90% → done |
| Brain coloring | Heterogeneous activation within each region (real model data) |
| Upload audio | Progress bar + brain updates |
| Image tab | Encoder dot visible (orange = not loaded, green = CLIP loaded) |
| Upload image | Prediction runs, brain shows visual-biased activation |
| Video tab | Unavailable overlay appears on top of dropzone |
| Toolbar ▶ TR | Appears after first prediction; animates 16 TR frames at ~5fps |
| Scrubber | Dragging jumps to any TR |
| Colorbar | Shows `[min, max] 20,484v` text |
| Pin A → run → Diff | Comparison panel + diff BOLD coloring works |

- [ ] **Step 4: Final commit if any loose changes**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git status
git add -A
git commit -m "fix: end-to-end verified — all inputs live, full vertex BOLD, temporal animation"
```
