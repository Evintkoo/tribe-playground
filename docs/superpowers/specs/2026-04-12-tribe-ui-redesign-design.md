# TRIBE v2 — UI Redesign Design Spec

**Date:** 2026-04-12

## Goal

Overhaul `tribe-v2-playground.html` to be simple, clean, and professional. The brain visualization is the hero — all controls are subordinate to it. No side panels, no menus, no icons, no gradients.

---

## Layout

```
┌─────────────────────────────────────────────────┐  44px
│  TRIBE v2  │  LATERAL  MEDIAL  DORSAL  │  ↓  ?  │  topbar
├─────────────────────────────────────────────────┤
│                                                 │
│  Visual  ●          [brain viewport]      + │   │  flex-1
│  Auditory ●                               │   │
│  Language ●                               - │   │
│  Prefrontal●                                │   │
│  Motor   ●                                  │   │
│  Parietal ●     [result card on predict]    │   │
│                                                 │
├─────────────────────────────────────────────────┤
│  TEXT  IMAGE  AUDIO                             │  bottom
│  ┌─────────────────────────── seq 16 │ Run ↵ ┐  │  bar
│  │ monospace input field…            │       │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Color Palette

No gradients anywhere except the BOLD colorbar (which encodes activation magnitude — must remain).

```css
--bg:      #0c0c0e;
--surface: #141416;
--overlay: #1a1a1d;
--border:  rgba(255,255,255,0.07);
--border2: rgba(255,255,255,0.12);

--t1: #f0f0f2;   /* primary text */
--t2: #888892;   /* secondary text */
--t3: #44444e;   /* tertiary / disabled */

--ac:    #5b7fff;              /* accent — flat, no gradient */
--ac-bg: rgba(91,127,255,0.1); /* accent background tint */

/* region colors */
--r-visual:     #e05252;
--r-auditory:   #e07a42;
--r-language:   #5b7fff;
--r-prefrontal: #3db870;
--r-motor:      #a855f7;
--r-parietal:   #d4a429;
```

Fonts: `Inter` (UI), `JetBrains Mono` (labels, input, data values).

---

## Components

### Topbar

Height: 44px. Background: `--bg`. Bottom border: `1px solid --border`.

- **Left:** `TRIBE` + `v2` in JetBrains Mono. `v2` is `--t3` color, slightly smaller (11px).
- **Divider:** 1px × 18px `--border` line.
- **Center (flex-1):** View chips — `LATERAL`, `MEDIAL`, `DORSAL`. Monospace, 10px, letter-spacing `.08em`. Active chip: `--t1` text, `1px solid --border2`, `--overlay` background, 20px border-radius. Inactive: `--t3`, no border, transparent bg.
- **Right:** Two icon buttons (28×28px, `--border` border, `--r` radius): Export (`↓`) and Info (`?`). Hover: `--overlay` bg, `--border2` border, `--t2` color.

### Brain Viewport

Fills all space between topbar and bottom bar. Three.js canvas renders to full size.

**ROI strip** — absolutely positioned, left edge, vertically centered:
- Each row: colored dot (7px circle) + uppercase label in JetBrains Mono 9px `--t2`.
- Row opacity `0.5` when toggled off, `1.0` when on. Click to toggle.
- No background panel — floats directly over the brain.

**BOLD colorbar** — absolutely positioned, right edge, vertically centered:
- 6px wide × 100px tall track. Colors: top `#e05252` → mid `rgba(255,255,255,0.3)` → bottom `#3db870`. Opacity 0.5.
- `+` label above, `−` label below. JetBrains Mono 9px `--t3`.

**Result card** — shown only after a prediction completes. Absolutely positioned, bottom-center of viewport (above the bottom bar). Hidden on load.
- Background: `--overlay`, border: `1px solid --border2`, radius: 8px, padding: `8px 16px`.
- Shows top activated regions (value + name) and inference time in ms.
- Values in JetBrains Mono 14px, colored by region. Labels in 9px `--t3` uppercase.
- If `demo_mode: true`, append a subtle `· demo` suffix in `--t3` after the ms value.

### Bottom Bar

Background: `--bg`. Top border: `1px solid --border`. Padding: `10px 16px 12px`.

**Mode tabs:**
- Three tabs: `TEXT`, `IMAGE`, `AUDIO`. JetBrains Mono, 10px, letter-spacing `.10em`.
- Active: `--t1`, 1.5px solid `--ac` bottom border, margin-bottom `-1px` to sit on the divider line.
- Inactive: `--t3`, no border, hover → `--t2`.
- Tab row has a `1px solid --border` bottom border acting as the underline baseline.

**Input row** (below tabs):
- Container: `background: --surface`, `border: 1px solid --border`, `border-radius: 8px`, `padding: 8px 12px`, `display: flex`, `align-items: center`, `gap: 10px`.

- **Text mode:** `<textarea>` (no border, no bg, transparent, resize:none, 1 row auto-expand). JetBrains Mono 12px, `--t2` color. Placeholder `--t3`.
- **Image / Audio mode:** Drop zone message when empty (`--t3`, dashed border on the container); when file attached, show a file chip above the input row:
  - Chip: `--overlay` bg, `1px solid --border2`, radius 6px, padding `5px 10px 5px 8px`.
  - Left: file extension badge (e.g. `JPG`) in `--ac-bg` bg, `--ac` text, mono 9px.
  - Middle: filename in `--t2`, size in `--t3`.
  - Right: `✕` remove button in `--t3`, hover `--t1`.

- **Seq control:** `seq 16` label in JetBrains Mono 10px `--t3`. Clicking cycles through 4 / 8 / 16 / 32 / 64.
- **Divider:** `1px solid --border`, 16px tall.
- **Run button:** `height: 28px`, `padding: 0 16px`, `background: --ac` (flat), `border-radius: 5px`, `font-size: 11px`, `font-weight: 600`, `color: #fff`. Keyboard shortcut badge `↵` in a tiny pill inside the button (9px, `rgba(255,255,255,0.2)` bg). While inferring: opacity 0.6, cursor wait. Cancel: border `1px solid rgba(224,82,82,0.3)`, text `--red`, transparent bg.

### Info Panel (? button)

Clicking `?` toggles a slide-in panel from the right (width 280px). Background: `--surface`, left border `1px solid --border`. Contains encoder status rows (loaded / not loaded) and download buttons for missing encoders.

Download button style: `--surface` bg, `1px solid --border2` border, flat. Progress bar under button when downloading: thin 2px track in `--border`, fill in `--ac`.

---

## What Is Removed

| Removed | Replaced by |
|---------|-------------|
| Left side panel (ROI toggles) | Floating ROI strip on brain edge |
| Right side panel (tabbed inputs) | Bottom bar |
| Topbar menus (File/View/Analysis/ROI/Export/Help) | View chips (center) + 2 icon buttons |
| Preset chips (VISUAL/LANGUAGE/AUDITORY…) | Removed entirely |
| Info modal | Slide-in panel via `?` button |
| Demo banner | `· demo` suffix on result card |
| Comparison panel | Removed (not in scope) |
| All gradients | Flat `--ac` color everywhere |

---

## What Is Preserved

- All Three.js brain visualization logic (mesh, vertex colors, camera controls)
- All WebSocket prediction logic
- All encoder loading / download logic (jobs, progress)
- All input handling (text, image, audio drag-drop, base64)
- ROI toggle state and coloring
- View switching (lateral/medial/dorsal)
- Keyboard shortcut: `Enter` (or `Cmd+Enter`) to run

---

## Non-Goals

- Mobile / responsive layout
- Dark/light mode toggle
- Comparison panel (pin A / compare B)
- Video encoder UI (V-JEPA2 not integrated)
- Animations or transitions beyond the existing Three.js ones
