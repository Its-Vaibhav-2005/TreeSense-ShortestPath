from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import heapq
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import imageio
from PIL import Image


Point = Tuple[int, int]


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
TEMPLATE_DIR = BASE_DIR / "templates"

for directory in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, TEMPLATE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Tree Sense Shortest Path", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@dataclass
class SolveResult:
    """Container for solve outputs."""

    image_id: str
    path: List[Point]
    explored: List[Point]
    summary_path: Path
    animation_path: Path
    animation_media_type: str
    settings: Dict[str, object]


def save_uploaded_image(upload: UploadFile) -> Tuple[str, Path, Tuple[int, int]]:
    """Persist an uploaded image as PNG and return its metadata."""

    suffix = (Path(upload.filename or "").suffix or "").lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported.")

    upload.file.seek(0)
    data = upload.file.read()
    upload.file.seek(0)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - PIL raises many types
        raise HTTPException(status_code=400, detail="Failed to decode image file.") from exc

    width, height = image.size
    image_id = uuid.uuid4().hex
    output_path = UPLOAD_DIR / f"{image_id}.png"
    image.save(output_path)

    meta = {
        "image_id": image_id,
        "filename": upload.filename,
        "stored_path": str(output_path),
        "width": width,
        "height": height,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }
    (UPLOAD_DIR / f"{image_id}.json").write_text(json.dumps(meta, indent=2))

    return image_id, output_path, (width, height)


def load_image_array(image_path: Path) -> np.ndarray:
    """Return an RGB numpy array for the stored image."""

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found on disk.")
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def compute_obstacle_grid(rgb: np.ndarray, dark_threshold: int) -> np.ndarray:
    """Create a boolean obstacle grid where True indicates blocked pixels."""

    grayscale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    return grayscale < float(dark_threshold)


def validate_point(point: Sequence[int], bounds: Tuple[int, int]) -> Point:
    """Ensure a point lies within the image bounds."""

    if len(point) != 2:
        raise ValueError("Point must be a 2-element sequence.")
    x, y = int(point[0]), int(point[1])
    width, height = bounds
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Point ({x}, {y}) is outside image bounds {width}x{height}.")
    return x, y


def xy_to_rc(point: Point) -> Point:
    """Convert an (x, y) point to (row, col)."""

    x, y = point
    return y, x


def rc_to_xy(point: Point) -> Point:
    """Convert a (row, col) point back to (x, y)."""

    row, col = point
    return col, row


def parse_int(value, default: int, name: str) -> int:
    """Convert a dynamic value to int or raise a validation error."""

    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer.") from exc


def find_nearest_free_cell(grid: np.ndarray, start: Point) -> Optional[Point]:
    """Return the closest free cell to `start` or None if the grid is fully blocked."""

    if not grid[start]:
        return start

    rows, cols = grid.shape
    q: deque[Point] = deque([start])
    visited: set[Point] = {start}

    # Allow 8-direction expansion to find the nearest free pixel quickly.
    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while q:
        cy, cx = q.popleft()
        for dy, dx in neighbor_offsets:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            neighbor = (ny, nx)
            if neighbor in visited:
                continue
            if not grid[neighbor]:
                return neighbor
            visited.add(neighbor)
            q.append(neighbor)

    return None


def movement_vectors(mode: str) -> List[Tuple[int, int, float]]:
    """Return movement offsets and their traversal cost."""

    if mode == "4":
        return [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
    if mode == "8":
        vectors = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        diag = [
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]
        return vectors + diag
    raise HTTPException(status_code=400, detail="neighbor_mode must be '4' or '8'.")


def heuristic_fn(name: str):
    """Return the requested heuristic function."""

    if name == "manhattan":
        return lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    if name == "euclidean":
        return lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    raise HTTPException(status_code=400, detail="heuristic must be 'manhattan' or 'euclidean'.")


def a_star(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    neighbor_mode: str,
    heuristic_name: str,
) -> Tuple[List[Point], List[Point]]:
    """Run A* search and return the best path and explored nodes order."""

    if grid[tuple(start)]:
        raise HTTPException(status_code=400, detail="Start lies on an obstacle.")
    if grid[tuple(goal)]:
        raise HTTPException(status_code=400, detail="Goal lies on an obstacle.")

    movements = movement_vectors(neighbor_mode)
    heuristic = heuristic_fn(heuristic_name)

    rows, cols = grid.shape
    g_costs: Dict[Point, float] = {start: 0.0}
    came_from: Dict[Point, Point] = {}
    open_heap: List[Tuple[float, int, Point]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    explored_order: List[Point] = []
    closed: set[Point] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        explored_order.append(current)

        if current == goal:
            path: List[Point] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, explored_order

        cy, cx = current
        current_cost = g_costs[current]
        for dy, dx, move_cost in movements:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            if grid[ny, nx]:
                continue
            neighbor = (ny, nx)
            tentative = current_cost + move_cost
            if tentative < g_costs.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_costs[neighbor] = tentative
                counter += 1
                score = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (score, counter, neighbor))

    return [], explored_order


def solve_with_waypoints(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    waypoints: Sequence[Point],
    neighbor_mode: str,
    heuristic_name: str,
) -> Tuple[List[Point], List[Point]]:
    """Solve sequentially through any supplied waypoints."""

    sequence = [start, *waypoints, goal]
    full_path: List[Point] = []
    explored_all: List[Point] = []

    for idx in range(len(sequence) - 1):
        seg_start = sequence[idx]
        seg_goal = sequence[idx + 1]
        segment_path, segment_explored = a_star(grid, seg_start, seg_goal, neighbor_mode, heuristic_name)
        if not segment_path:
            raise HTTPException(
                status_code=400,
                detail=f"No path found between waypoint {idx} and {idx + 1}.",
            )
        if not full_path:
            full_path.extend(segment_path)
        else:
            full_path.extend(segment_path[1:])
        explored_all.extend(segment_explored)

    return full_path, explored_all


def ensure_ffmpeg() -> str:
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_path:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg executable not available. Install ffmpeg or add imageio-ffmpeg.",
            )
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    return ffmpeg_path


def compose_animation_frame(
    base: np.ndarray,
    obstacle_mask: np.ndarray,
    visited_mask: np.ndarray,
    path_mask: np.ndarray,
    start_rc: Point,
    goal_rc: Point,
) -> np.ndarray:
    """Build a single animation frame."""

    frame = base.copy()
    if obstacle_mask.any():
        frame[obstacle_mask] = (
            0.3 * frame[obstacle_mask].astype(np.float32)
            + 0.7 * np.array([30, 30, 30], dtype=np.float32)
        ).astype(np.uint8)
    frame[visited_mask] = [255, 165, 0]
    frame[path_mask] = [220, 20, 60]
    sy, sx = start_rc
    gy, gx = goal_rc
    frame[sy, sx] = [255, 0, 255]
    frame[gy, gx] = [0, 255, 255]
    return frame


def generate_summary_figure(
    rgb: np.ndarray,
    obstacle_grid: np.ndarray,
    path: Sequence[Point],
    explored: Sequence[Point],
    summary_path: Path,
) -> None:
    """Create and store the summary subplot figure."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(obstacle_grid, cmap="gray")
    axes[0, 1].set_title("Obstacle Grid (True = blocked)")
    axes[0, 1].axis("off")

    if path:
        rows = [pt[0] for pt in path]
        cols = [pt[1] for pt in path]
    else:
        rows, cols = [], []

    axes[1, 0].imshow(rgb)
    if rows:
        axes[1, 0].plot(cols, rows, color="red", linewidth=3, alpha=0.9)
    axes[1, 0].set_title("Final Path Overlay")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylim(rgb.shape[0], 0)

    axes[1, 1].imshow(rgb)
    if explored:
        y_vals = [pt[0] for pt in explored]
        x_vals = [pt[1] for pt in explored]
        axes[1, 1].scatter(x_vals, y_vals, s=2, c="gold", alpha=0.35, label="Explored")
    if rows:
        axes[1, 1].plot(cols, rows, color="red", linewidth=3, alpha=0.9, label="Path")
    axes[1, 1].set_title("Explored Nodes vs Path")
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    if explored or rows:
        axes[1, 1].legend(loc="upper right")

    fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.94, bottom=0.04)
    fig.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Add near the imports or just above generate_animation
from math import ceil

# Tunables. Keep small to avoid stalls on large grids.
MAX_VIDEO_FRAMES = 1200      # hard cap including tail frames
MAX_GIF_FRAMES = 300         # hard cap for GIF fallback
TAIL_FRAMES = 15             # frames after exploration finishes
FPS = 20                     # keep modest to limit work

def generate_animation(
    rgb: np.ndarray,
    obstacle_grid: np.ndarray,
    path: Sequence[Point],
    explored: Sequence[Point],
    start_rc: Point,
    goal_rc: Point,
    animation_path: Path,
) -> Tuple[Path, str]:
    """
    Render the search animation and return (path, media_type).

    Changes:
    - Stream frames to encoder (no giant in-memory list).
    - Downsample explored frames if needed to respect caps.
    - Prefer fast, broadly-available codecs.
    - Constrained GIF fallback with subsampling.
    """

    ensure_ffmpeg()
    animation_path.parent.mkdir(parents=True, exist_ok=True)

    # Precompute masks once
    obstacle_mask = obstacle_grid.astype(bool)
    visited_mask = np.zeros_like(obstacle_grid, dtype=bool)
    path_mask = np.zeros_like(obstacle_grid, dtype=bool)
    if path:
        coords = np.asarray(path, dtype=int)
        path_mask[coords[:, 0], coords[:, 1]] = True

    base = rgb.copy()

    # Deduplicate explored nodes while preserving order
    unique_explored: List[Point] = []
    seen: set[Point] = set()
    for node in explored:
        if node not in seen:
            seen.add(node)
            unique_explored.append(node)

    # Throttle frame count
    max_explore_frames = max(0, MAX_VIDEO_FRAMES - TAIL_FRAMES)
    explore_len = len(unique_explored)
    stride = 1 if explore_len <= max_explore_frames else ceil(explore_len / max_explore_frames)

    # Helper: one frame composition
    def compose(idx_mark: Optional[int]) -> np.ndarray:
        if idx_mark is not None:
            y, x = unique_explored[idx_mark]
            visited_mask[y, x] = True
        frame = compose_animation_frame(
            base,
            obstacle_mask,
            visited_mask,
            path_mask,
            start_rc,
            goal_rc,
        )
        return np.ascontiguousarray(frame, dtype=np.uint8)

    # Try MP4 first with a fast codec
    attempts = [
        (animation_path.with_suffix(".mp4"), "mpeg4", "video/mp4", {"fps": FPS, "quality": 7}),
        (animation_path.with_suffix(".webm"), "libvpx-vp9", "video/webm", {"fps": FPS}),
    ]

    last_error: Optional[Exception] = None
    for target, codec, media_type, extra in attempts:
        try:
            with imageio.get_writer(
                str(target),
                format="FFMPEG",
                mode="I",
                codec=codec,
                fps=extra.get("fps", FPS),
                # Avoid slow pixel conversions when possible
                output_params=["-pix_fmt", "yuv420p"] if media_type == "video/mp4" else None,
                quality=extra.get("quality", None),
            ) as writer:
                # Stream exploration frames with throttling
                for i in range(0, explore_len, stride):
                    writer.append_data(compose(i))
                # Tail frames to hold the final view
                for _ in range(TAIL_FRAMES):
                    writer.append_data(compose(None))
            return target, media_type
        except Exception as exc:
            last_error = exc

    # GIF fallback with strict cap and subsampling
    try:
        gif_path = animation_path.with_suffix(".gif")
        max_explore_frames = max(0, min(MAX_GIF_FRAMES - TAIL_FRAMES, MAX_GIF_FRAMES))
        stride = 1 if explore_len <= max_explore_frames else ceil(explore_len / max_explore_frames)
        frames: List[np.ndarray] = []
        for i in range(0, explore_len, stride):
            frames.append(compose(i))
        for _ in range(TAIL_FRAMES):
            frames.append(compose(None))
        # Keep GIF under control
        imageio.mimsave(str(gif_path), frames, format="GIF", duration=1.0 / FPS)
        return gif_path, "image/gif"
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError("Failed to encode animation with available codecs.") from last_error
        raise RuntimeError("Failed to encode animation and GIF fallback.") from exc

def store_result_metadata(result: SolveResult) -> Path:
    """Persist metadata for later retrieval."""

    meta = {
        "image_id": result.image_id,
        "path": [list(rc_to_xy(pt)) for pt in result.path],
        "explored": [list(rc_to_xy(pt)) for pt in result.explored],
        "summary_path": result.summary_path.name,
        "animation_path": result.animation_path.name,
        "animation_media_type": result.animation_media_type,
        "settings": result.settings,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = RESULT_DIR / result.image_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def load_result_metadata(image_id: str) -> Dict[str, object]:
    """Load stored metadata if available."""

    meta_path = RESULT_DIR / image_id / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Result not found. Run /solve first.")
    return json.loads(meta_path.read_text())


def process_solve_request(
    image_id: str,
    start_xy: Point,
    goal_xy: Point,
    waypoints_xy: Sequence[Point],
    obstacle_points_xy: Sequence[Point],
    dark_threshold: int,
    neighbor_mode: str,
    heuristic_name: str,
) -> SolveResult:
    """Execute the full pipeline for the provided parameters."""

    if not (0 <= dark_threshold <= 255):
        raise HTTPException(status_code=400, detail="dark_threshold must be between 0 and 255.")

    image_path = UPLOAD_DIR / f"{image_id}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Unknown image_id.")

    rgb = load_image_array(image_path)
    height, width = rgb.shape[0], rgb.shape[1]
    bounds = (width, height)

    start_xy = validate_point(start_xy, bounds)
    goal_xy = validate_point(goal_xy, bounds)
    waypoints_xy = [validate_point(pt, bounds) for pt in waypoints_xy]
    obstacle_points_xy = [validate_point(pt, bounds) for pt in obstacle_points_xy]

    obstacle_grid = compute_obstacle_grid(rgb, dark_threshold)
    for ox, oy in obstacle_points_xy:
        obstacle_grid[oy, ox] = True

    start_rc = xy_to_rc(start_xy)
    goal_rc = xy_to_rc(goal_xy)
    waypoint_rc = [xy_to_rc(pt) for pt in waypoints_xy]

    start_adjusted = False
    goal_adjusted = False
    waypoint_adjustments: List[Dict[str, object]] = []

    if obstacle_grid[start_rc]:
        nearest = find_nearest_free_cell(obstacle_grid, start_rc)
        if nearest is None:
            raise HTTPException(status_code=400, detail="Start lies on an obstacle and no nearby free cell found.")
        start_rc = nearest
        start_adjusted = True

    if obstacle_grid[goal_rc]:
        nearest = find_nearest_free_cell(obstacle_grid, goal_rc)
        if nearest is None:
            raise HTTPException(status_code=400, detail="Goal lies on an obstacle and no nearby free cell found.")
        goal_rc = nearest
        goal_adjusted = True

    for idx, wpt in enumerate(list(waypoint_rc)):
        if not obstacle_grid[wpt]:
            continue
        nearest = find_nearest_free_cell(obstacle_grid, wpt)
        if nearest is None:
            raise HTTPException(
                status_code=400,
                detail=f"Waypoint {idx + 1} lies on an obstacle and no nearby free cell found.",
            )
        waypoint_adjustments.append(
            {
                "index": idx + 1,
                "original": list(rc_to_xy(wpt)),
                "adjusted": list(rc_to_xy(nearest)),
            }
        )
        waypoint_rc[idx] = nearest

    path_rc, explored_rc = solve_with_waypoints(
        obstacle_grid,
        start_rc,
        goal_rc,
        waypoint_rc,
        neighbor_mode,
        heuristic_name,
    )

    if not path_rc:
        raise HTTPException(status_code=400, detail="No path found with the provided parameters.")

    result_dir = RESULT_DIR / image_id
    summary_path = result_dir / "summary.png"
    animation_base = result_dir / "search"

    generate_summary_figure(rgb, obstacle_grid, path_rc, explored_rc, summary_path)
    final_animation_path, media_type = generate_animation(
        rgb,
        obstacle_grid,
        path_rc,
        explored_rc,
        start_rc,
        goal_rc,
        animation_base,
    )

    result = SolveResult(
        image_id=image_id,
        path=path_rc,
        explored=explored_rc,
        summary_path=summary_path,
        animation_path=final_animation_path,
        animation_media_type=media_type,
        settings={
            "dark_threshold": dark_threshold,
            "neighbor_mode": neighbor_mode,
            "heuristic": heuristic_name,
            "waypoint_count": len(waypoint_rc),
            "obstacle_count": len(obstacle_points_xy),
            "start_adjusted": start_adjusted,
            "goal_adjusted": goal_adjusted,
            "waypoint_adjustments": waypoint_adjustments,
        },
    )
    store_result_metadata(result)
    return result


def parse_coords_payload(raw: Dict[str, object]) -> Tuple[Point, Point, List[Point], List[Point]]:
    """Extract coordinate collections from payload dictionaries."""

    try:
        start = tuple(raw["start"])  # type: ignore[arg-type]
        goal = tuple(raw["goal"])  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="start and goal coordinates are required.") from exc

    waypoints = [tuple(item) for item in raw.get("waypoints", [])]  # type: ignore[list-item]
    obstacles = [tuple(item) for item in raw.get("obstacles", [])]  # type: ignore[list-item]

    return start, goal, waypoints, obstacles


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request) -> HTMLResponse:
    """Render the upload form."""

    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)) -> RedirectResponse:
    """Accept an image upload and redirect to the selection UI."""

    image_id, _, dimensions = save_uploaded_image(file)
    redirect_url = request.url_for("select_points", image_id=image_id)
    response = RedirectResponse(url=str(redirect_url), status_code=303)
    response.set_cookie("image_dims", json.dumps({"width": dimensions[0], "height": dimensions[1]}), max_age=3600)
    return response


@app.get("/select/{image_id}", response_class=HTMLResponse)
async def select_points(request: Request, image_id: str) -> HTMLResponse:
    """Render the point-selection UI for the uploaded image."""

    image_path = UPLOAD_DIR / f"{image_id}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found. Upload again.")

    try:
        meta = json.loads((UPLOAD_DIR / f"{image_id}.json").read_text())
    except Exception:
        image = Image.open(image_path)
        meta = {"width": image.width, "height": image.height}

    context = {
        "request": request,
        "image_id": image_id,
        "image_url": request.url_for("get_uploaded_image", image_id=image_id),
        "width": meta.get("width"),
        "height": meta.get("height"),
    }
    return templates.TemplateResponse("select.html", context)


@app.get("/uploads/{image_id}.png")
async def get_uploaded_image(image_id: str) -> FileResponse:
    """Serve the stored upload."""

    image_path = UPLOAD_DIR / f"{image_id}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path)


@app.post("/solve")
async def solve_endpoint(request: Request) -> Response:
    """Solve via either JSON payload or HTML form submission."""

    content_type = request.headers.get("content-type", "")
    is_json = content_type.startswith("application/json")

    if is_json:
        payload = await request.json()
        image_id = str(payload.get("image_id", "")).strip()
        if not image_id:
            raise HTTPException(status_code=400, detail="image_id is required in JSON payload.")
        start_xy, goal_xy, waypoints_xy, obstacles_xy = parse_coords_payload(payload)
        dark_threshold = parse_int(payload.get("dark_threshold"), 80, "dark_threshold")
        neighbor_mode = str(payload.get("neighbor_mode", "8")).strip()
        heuristic_name = str(payload.get("heuristic", "euclidean")).strip().lower()
        result = process_solve_request(
            image_id=image_id,
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoints_xy=waypoints_xy,
            obstacle_points_xy=obstacles_xy,
            dark_threshold=dark_threshold,
            neighbor_mode=neighbor_mode,
            heuristic_name=heuristic_name,
        )

        summary_url = request.url_for("result_summary", image_id=result.image_id)
        animation_url = request.url_for("result_animation", image_id=result.image_id)
        return JSONResponse(
            {
                "image_id": result.image_id,
                "path": [list(rc_to_xy(pt)) for pt in result.path],
                "explored": [list(rc_to_xy(pt)) for pt in result.explored],
                "settings": result.settings,
                "summary_url": str(summary_url),
                "animation_url": str(animation_url),
                "animation_media_type": result.animation_media_type,
            }
        )

    form = await request.form()
    image_id = str(form.get("image_id", "")).strip()
    if not image_id:
        raise HTTPException(status_code=400, detail="image_id missing from form submission.")
    coords_raw = form.get("coords_json")
    if not coords_raw:
        raise HTTPException(status_code=400, detail="coords_json missing from form submission.")
    try:
        coords_payload = json.loads(coords_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid coords_json payload.") from exc

    start_xy, goal_xy, waypoints_xy, obstacles_xy = parse_coords_payload(coords_payload)

    result = process_solve_request(
        image_id=image_id,
        start_xy=start_xy,
        goal_xy=goal_xy,
        waypoints_xy=waypoints_xy,
        obstacle_points_xy=obstacles_xy,
        dark_threshold=parse_int(form.get("dark_threshold"), 80, "dark_threshold"),
        neighbor_mode=str(form.get("neighbor_mode", "8")).strip(),
        heuristic_name=str(form.get("heuristic", "euclidean")).strip().lower(),
    )

    summary_url = request.url_for("result_summary", image_id=result.image_id)
    animation_url = request.url_for("result_animation", image_id=result.image_id)

    context = {
        "request": request,
        "image_id": result.image_id,
        "summary_url": summary_url,
        "animation_url": animation_url,
        "animation_media_type": result.animation_media_type,
        "path_length": len(result.path),
        "explored_count": len(result.explored),
        "path_points": [rc_to_xy(pt) for pt in result.path],
        "settings": result.settings,
    }
    return templates.TemplateResponse("result.html", context)


@app.get("/result/{image_id}", response_class=HTMLResponse)
async def result_page(request: Request, image_id: str) -> HTMLResponse:
    """Render a previously computed result page."""

    meta = load_result_metadata(image_id)
    summary_url = request.url_for("result_summary", image_id=image_id)
    animation_url = request.url_for("result_animation", image_id=image_id)

    context = {
        "request": request,
        "image_id": image_id,
        "summary_url": summary_url,
        "animation_url": animation_url,
        "animation_media_type": meta.get("animation_media_type", "video/mp4"),
        "path_length": len(meta.get("path", [])),
        "explored_count": len(meta.get("explored", [])),
        "path_points": meta.get("path", []),
        "settings": meta.get("settings", {}),
    }
    return templates.TemplateResponse("result.html", context)


@app.get("/result/{image_id}/summary.png")
async def result_summary(image_id: str) -> FileResponse:
    """Serve the summary figure."""

    summary_path = RESULT_DIR / image_id / "summary.png"
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="Summary image not found.")
    return FileResponse(summary_path, media_type="image/png")


@app.get("/result/{image_id}/animation")
async def result_animation(image_id: str) -> FileResponse:
    """Serve the search animation."""

    meta = load_result_metadata(image_id)
    filename = meta.get("animation_path")
    if not filename:
        raise HTTPException(status_code=404, detail="Animation not found.")

    file_path = RESULT_DIR / image_id / str(filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Animation not found.")

    media_type = meta.get("animation_media_type")
    if not isinstance(media_type, str):
        suffix_map = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".gif": "image/gif",
        }
        media_type = suffix_map.get(file_path.suffix.lower(), "application/octet-stream")

    return FileResponse(file_path, media_type=media_type)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):  # type: ignore[override]
    """Return JSON errors for API clients and fallback HTML for forms."""

    accepts = request.headers.get("accept", "") or ""
    wants_json = "application/json" in accepts
    if wants_json:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "status_code": exc.status_code, "message": exc.detail},
        status_code=exc.status_code,
    )


def create_error_template() -> None:
    """Ensure a minimal error template exists for graceful fallbacks."""

    error_template = TEMPLATE_DIR / "error.html"
    if error_template.is_file():
        return
    error_template.write_text(
        """<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Error</title>"""
        """<style>body{font-family:Arial,Helvetica,sans-serif;margin:3rem;max-width:40rem;}"""
        """h1{color:#b00020;}a{color:#005a9c;text-decoration:none;}</style></head><body>"""
        """<h1>{{ status_code }}</h1><p>{{ message }}</p><p><a href=\"/\">Back to upload</a></p></body></html>"""
    )


create_error_template()
