# Tree Sense Shortest Path API

This project exposes a FastAPI service that uploads an image, allows you to choose start/goal/waypoint/obstacle pixels, converts the image into a traversable grid, and runs configurable A* pathfinding. It produces both a static summary figure and an animation of the search.

## Quickstart

1. Create a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # use .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Launch the API with auto-reload during development:

   ```bash
   uvicorn main:app --reload
   ```

3. Open `http://127.0.0.1:8000/` in your browser. Upload a PNG or JPG, pick your points, adjust thresholds if needed, and run the search. The result page displays the summary PNG and the animation, with links to download both outputs.

> **Note:** Generating MP4 output requires `ffmpeg`. If it is not installed globally, the app falls back to the bundled binary from the `imageio-ffmpeg` package, so no manual setup is usually needed.

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload form. |
| `/upload` | POST | Accepts an image file (`multipart/form-data`). Redirects to `/select/{image_id}`. |
| `/select/{image_id}` | GET | Interactive point-selection UI in the browser. |
| `/solve` | POST | Accepts either form submissions (from the UI) or JSON payloads to trigger A* search. |
| `/result/{image_id}` | GET | Result page with embedded outputs. |
| `/result/{image_id}/summary.png` | GET | Download the Matplotlib subplot PNG. |
| `/result/{image_id}/animation` | GET | Download the animation (MP4 or WebM). |

### JSON Solve Payload

Send a `POST /solve` request with `Content-Type: application/json` to compute a path programmatically:

```json
{
  "image_id": "<returned-from-upload>",
  "start": [120, 340],
  "goal": [760, 120],
  "waypoints": [[300, 280]],
  "obstacles": [[420, 260]],
  "dark_threshold": 80,
  "neighbor_mode": "8",
  "heuristic": "euclidean"
}
```

Successful responses include the ordered path, explored nodes, and URLs to download the summary and animation files.

## Development Notes

- All processing happens synchronously within the request as required.
- Dark pixels (grayscale below `dark_threshold`) are obstacles; you can also manually mark per-pixel obstacles.
- Choose 4- or 8-neighbor movement and Manhattan or Euclidean heuristic at solve time.
- Intermediate artifacts are stored under `data/` with per-image subdirectories.
- If your selected start/goal or any waypoint lands on a blocked pixel the solver nudges it to the closest free pixel and records that adjustment in the result metadata.
