# Frontend

The frontend now behaves like a release-style live ASL recognition client rather than a test dashboard.

## Visible product flow

- compact live recognition layout with webcam left and prediction right on desktop
- short top status cards for:
  - backend connection
  - mode
  - session
- default webcam keypoint overlay
- user-facing controls:
  - `Start Camera`
  - `Stop Camera`
  - `Show Keypoints`
  - `Start Recognition`
  - `Stop Recognition`
  - `Reset`
- stabilized recognition display as the main output
- raw Top-K and manual single-frame capture moved under `Advanced details`

## Required backend URL

Create a local environment file from the example and point it to the FastAPI backend:

```powershell
Copy-Item .env.example .env
```

Expected value:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Setup

From the `frontend/` directory:

```powershell
npm install
```

## Run

From the `frontend/` directory:

```powershell
npm run dev
```

The Vite dev server runs at `http://127.0.0.1:5173`.

## Backend requirement

The FastAPI backend must already be running on `http://127.0.0.1:8000` for the dashboard to load health data, preview keypoints, and run real recognition.

Run both services:

```powershell
# terminal 1, from repo root
uvicorn app.main:app --app-dir backend --reload

# terminal 2, from frontend/
npm run dev
```

## Browser webcam permission

When you click `Start Camera`, the browser will ask for webcam permission. You must allow access for the Phase 4D webcam capture flow to work.

The UI handles common camera states such as:

- permission denied
- no camera found
- camera already in use
- capture not ready
- raw session warming up
- no landmarks
- collecting votes
- held confusion
- low confidence
- stabilized output
- backend request failure

## Continuous real inference

The frontend now supports:

- `Start Camera`
- `Stop Camera`
- `Show Keypoints`
- `Start Recognition`
- `Stop Recognition`
- `Reset`

The live capture loop is now fixed to the fastest validated setting:

- `100ms` per frame
- about `10 FPS`
- capture payload tuned for lower latency:
  - width `480px`
  - `image/jpeg`
  - quality `0.6`

Only one request is allowed in flight at a time. If a frame request is still running, the next timer tick is skipped instead of queueing overlapping requests.

## Background buffer warming

- `Start Camera` now begins background frame submission immediately.
- The backend starts filling and maintaining the rolling `30`-frame buffer as soon as the camera is on.
- `Start Recognition` no longer begins from zero if the background buffer is already warm.
- If enough valid frames are already buffered, recognition can begin using the hot buffer immediately.
- `Stop Recognition` pauses user-facing recognition while the camera keeps the background buffer warm.
- `Stop Camera` stops the loop, releases the webcam, and clears the backend session.
- `Reset` clears the current session state and starts buffer warming from zero again if the camera remains on.
- brief hand loss now uses a time-based grace window of about `2.0s` before the backend clears the stale session

## Keypoint overlay

- Keypoints are shown by default.
- The visible webcam overlay now runs in the browser with MediaPipe Tasks hand landmark detection.
- The overlay updates from the live video element instead of waiting for backend responses.
- The current overlay is hand-only for lower latency and lighter browser load.
- Backend MediaPipe preprocessing still remains the source of truth for model inference.
- `POST /api/inference/frame-debug` remains available for backend debugging, but it is not used during normal live camera use.

## Runtime behavior

- The main recognition output now prefers stabilized signs only.
- After a stable sign is accepted, the backend briefly holds that accepted output during short transitions instead of immediately swapping to weak random guesses.
- If hands disappear:
  - the system first enters `holding_context`
  - then returns to `waiting_for_hands`
  - and clears stale sequence state after the grace window

## What the browser now does

- open the webcam
- show a live preview with keypoints
- send frames continuously to the real recognition endpoint
- warm up the 30-frame rolling session buffer as soon as the camera starts
- display stabilized recognition as the main output
- keep raw Top-K as secondary information under `Advanced details`
- avoid spamming accepted words during sign transitions

## Live dashboard layout

- desktop:
  - larger left column for webcam preview and recognition controls
  - right column for sticky live recognition output
- mobile:
  - webcam and prediction stack vertically
  - the prediction card stays directly under the webcam section

## Coming later

Future phases can still add richer transport and deployment support, but the current UI is intentionally focused on isolated live sign recognition.

Not included yet:

- authentication
- WebSockets
