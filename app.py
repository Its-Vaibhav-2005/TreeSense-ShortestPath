from __future__ import annotations

import asyncio
import heapq
import io
import json
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse, FileResponse
from matplotlib.animation import FFMpegWriter
from PIL import Image

# ---------------- Types ----------------
Point = Tuple[int, int]

# ---------------- I/O Helpers ----------------
def loadImage(path: str, maxDim: int = 900) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    scale = min(maxDim / width, maxDim / height, 1.0)
    if scale < 1.0:
        newSize = (int(width * scale), int(height * scale))
        try:
            img = img.resize(newSize, Image.Resampling.LANCZOS)
        except AttributeError:
            img = img.resize(newSize, Image.LANCZOS)
    return np.array(img)

def computeTreeMask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    maxRb = np.maximum(r, b)
    exg = (g - maxRb) / (g + maxRb + 1e-6)
    return (exg > threshold).astype(np.int32)

def buildCostMap(treeMask: np.ndarray, treeCost: int = 20, openCost: int = 1) -> np.ndarray:
    return np.where(treeMask == 1, treeCost, openCost).astype(np.int32)

def ndarrayToPngBytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# ---------------- A* Animator ----------------
class AStarAnimator:
    def __init__(
        self,
        rgb: np.ndarray,
        costMap: np.ndarray,
        start: Point,
        goal: Point,
        expansionsPerFrame: int = 10,
    ) -> None:
        self.rgb = rgb
        self.cost = costMap
        self.start = start
        self.goal = goal
        self.expansionsPerFrame = max(1, expansionsPerFrame)

        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.g = {start: 0}
        self.f = {start: self._heuristic(start, goal)}
        self.parents: dict[Point, Point] = {}
        self.open: List[Tuple[float, Point]] = [(self.f[start], start)]
        self.closed: set[Point] = set()
        self.done = False
        self.path: List[Point] = []
        self.lastCurrent: Optional[Point] = None

        self.base = rgb.copy()
        mask = costMap > costMap.min()
        greenTint = np.zeros_like(self.base)
        greenTint[:] = [0, 140, 0]
        self.base[mask] = (
            0.4 * self.base[mask].astype(np.float32)
            + 0.6 * greenTint[mask].astype(np.float32)
        ).astype(np.uint8)

        self.frameBuffer = np.empty_like(self.base)
        self.visitedMask = np.zeros(costMap.shape, dtype=bool)
        self.frontierMask = np.zeros(costMap.shape, dtype=bool)

    @staticmethod
    def _heuristic(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _advance(self) -> None:
        for _ in range(self.expansionsPerFrame):
            if self.done:
                return
            if not self.open:
                self.done = True
                return

            _, current = heapq.heappop(self.open)
            if current in self.closed:
                continue

            self.closed.add(current)
            self.visitedMask[current] = True
            self.lastCurrent = current

            if current == self.goal:
                path: List[Point] = [current]
                while current in self.parents:
                    current = self.parents[current]
                    path.append(current)
                self.path = list(reversed(path))
                self.done = True
                return

            currentCost = self.g[current]
            for dy, dx in self.neighbors:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
                    tentative = currentCost + int(self.cost[ny, nx])
                    neighbor = (ny, nx)
                    if tentative < self.g.get(neighbor, float("inf")):
                        self.parents[neighbor] = current
                        self.g[neighbor] = tentative
                        self.f[neighbor] = tentative + self._heuristic(neighbor, self.goal)
                        heapq.heappush(self.open, (self.f[neighbor], neighbor))

    def step(self) -> np.ndarray:
        self._advance()
        np.copyto(self.frameBuffer, self.base)

        if self.visitedMask.any():
            self.frameBuffer[self.visitedMask] = [100, 149, 237]

        self.frontierMask.fill(False)
        for _, node in self.open:
            if node not in self.closed:
                self.frontierMask[node] = True
        if self.frontierMask.any():
            self.frameBuffer[self.frontierMask] = [255, 165, 0]

        if self.lastCurrent is not None and not self.done:
            cy, cx = self.lastCurrent
            self.frameBuffer[cy, cx] = [65, 105, 225]

        sy, sx = self.start
        gy, gx = self.goal
        self.frameBuffer[sy, sx] = [255, 0, 255]
        self.frameBuffer[gy, gx] = [255, 255, 0]

        if self.path:
            for py, px in self.path:
                self.frameBuffer[py, px] = [255, 0, 0]
        return self.frameBuffer

def plotPathsSummary(anim: AStarAnimator, rgb: np.ndarray, original: np.ndarray, costMap: np.ndarray):
    if not anim.path:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows = [pt[0] for pt in anim.path]
    cols = [pt[1] for pt in anim.path]

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    costVis = costMap.astype(np.float32)
    axes[0, 1].imshow(costVis, cmap="viridis")
    axes[0, 1].set_title("Cost Map (Higher = Trees)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb)
    axes[1, 0].plot(cols, rows, color="red", linewidth=3, alpha=0.9)
    axes[1, 0].set_title("Best Path Overlay")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylim(rgb.shape[0], 0)

    axes[1, 1].imshow(rgb)
    exploredY, exploredX = np.where(anim.visitedMask)
    if exploredY.size:
        axes[1, 1].scatter(exploredX, exploredY, s=2, c="gold", alpha=0.3, label="Explored")
    axes[1, 1].plot(cols, rows, color="red", linewidth=3, alpha=0.9, label="Best Path")
    axes[1, 1].set_title("Explored vs Best Path")
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    axes[1, 1].legend(loc="upper right")

    fig.subplots_adjust(hspace=0.095, wspace=0.062, top=0.945, bottom=0.015, left=0.010, right=0.990)
    return fig

def fallbackStraightPath(start: Point, goal: Point) -> List[Point]:
    y0, x0 = start
    y1, x1 = goal
    path: List[Point] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        path.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return path

def resolveFfmpegPath() -> str:
    ffmpegPath = plt.rcParams.get("animation.ffmpeg_path")
    if ffmpegPath and Path(ffmpegPath).is_file():
        return ffmpegPath
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or install imageio-ffmpeg."
        ) from exc

# ---------------- Job Model ----------------
@dataclass
class Job:
    jobId: str
    uploadPath: Path
    outputRoot: Path
    rgb: Optional[np.ndarray] = None
    originalRgb: Optional[np.ndarray] = None
    treeMask: Optional[np.ndarray] = None
    costMap: Optional[np.ndarray] = None
    start: Optional[Point] = None
    goal: Optional[Point] = None
    animator: Optional[AStarAnimator] = None
    finalVideoPath: Optional[Path] = None
    summaryImagePath: Optional[Path] = None
    error: Optional[str] = None
    done: bool = False
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    currentFrame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {msg}")

# ---------------- Job Store ----------------
class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, uploadPath: Path) -> Job:
        jobId = uuid.uuid4().hex[:12]
        outRoot = Path("BestPath-JungleMode") / jobId
        job = Job(jobId=jobId, uploadPath=uploadPath, outputRoot=outRoot)
        with self._lock:
            self._jobs[jobId] = job
        return job

    def get(self, jobId: str) -> Job:
        with self._lock:
            if jobId not in self._jobs:
                raise KeyError("Job not found")
            return self._jobs[jobId]

JOB_STORE = JobStore()

# ---------------- Processing ----------------
def runPipeline(job: Job, expansionsPerFrame: int = 25, intervalMs: int = 12) -> None:
    try:
        job.log("Loading image")
        rgb = loadImage(str(job.uploadPath), maxDim=900)
        job.rgb = rgb
        job.originalRgb = rgb.copy()
        job.currentFrame = rgb.copy()

        job.log("Computing vegetation mask")
        treeMask = computeTreeMask(rgb, threshold=0.15)
        job.treeMask = treeMask

        job.log("Building traversal cost map")
        costMap = buildCostMap(treeMask, treeCost=20, openCost=1)
        job.costMap = costMap

        if job.start is None or job.goal is None:
            job.error = "Start and Goal not set"
            job.log("Error: start and goal not set")
            return

        job.log(f"Starting A* search from {job.start} to {job.goal}")
        animator = AStarAnimator(rgb, costMap, job.start, job.goal, expansionsPerFrame=expansionsPerFrame)
        job.animator = animator

        videoDir = job.outputRoot / "video"
        imageDir = job.outputRoot / "image"
        videoDir.mkdir(parents=True, exist_ok=True)
        imageDir.mkdir(parents=True, exist_ok=True)
        videoPath = videoDir / "search_animation.mp4"

        # Resolve ffmpeg
        try:
            resolved = resolveFfmpegPath()
            plt.rcParams["animation.ffmpeg_path"] = resolved
            writer = FFMpegWriter(fps=max(1, int(round(1000 / intervalMs))), codec="libx264", bitrate=1800)
            job.log("ffmpeg resolved and writer ready")
        except Exception as exc:
            job.log(f"Video disabled: {exc}")
            writer = None  # type: ignore

        # Save animation with live frame tap
        fig, ax = plt.subplots(figsize=(8, 8 * animator.rgb.shape[0] / animator.rgb.shape[1]))
        ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
        ax.axis("off")
        frame = animator.step()
        im = ax.imshow(frame)
        job.currentFrame = frame.copy()

        try:
            if writer is not None:
                with writer.saving(fig, str(videoPath), 120):
                    writer.grab_frame()
                    job.log("Animation frame 1 recorded")
                    while not animator.done:
                        frame = animator.step()
                        im.set_data(frame)
                        writer.grab_frame()
                        job.currentFrame = frame.copy()
                    for _ in range(15):
                        writer.grab_frame()
                job.finalVideoPath = videoPath
                job.log(f"Video saved: {videoPath}")
            else:
                while not animator.done:
                    frame = animator.step()
                    im.set_data(frame)
                    job.currentFrame = frame.copy()
        except Exception as exc:
            job.log(f"Animation failed: {exc}")
        finally:
            plt.close(fig)

        if animator.path:
            treesCrossed = int(np.sum(treeMask[tuple(np.array(animator.path).T)]))
            job.log(f"Path length: {len(animator.path)}")
            job.log(f"Estimated trees crossed: {treesCrossed}")

            job.log("Creating summary figure")
            fig2 = plotPathsSummary(animator, animator.base, job.originalRgb, costMap)
            if fig2 is not None:
                summaryPath = job.outputRoot / "image" / "best_path_summary.png"
                fig2.savefig(summaryPath, bbox_inches="tight", dpi=160)
                plt.close(fig2)
                job.summaryImagePath = summaryPath
                job.log(f"Summary saved: {summaryPath}")
        else:
            job.log("No viable path found. Generating straight-line fallback")
            direct = fallbackStraightPath(job.start, job.goal)
            treesCrossed = int(np.sum(treeMask[tuple(np.array(direct).T)]))
            job.log(f"Direct path length: {len(direct)}")
            job.log(f"Estimated trees encountered on direct path: {treesCrossed}")

        job.done = True
        job.log("Processing complete")
    except Exception as exc:
        job.error = str(exc)
        job.done = True
        job.log(f"Fatal error: {exc}")

# ---------------- FastAPI App ----------------
app = FastAPI(title="Jungle Path A* — Minimal UI")

INDEX_HTML = """<!doctype html>
<html>
  <body>
    <h2>Upload Image</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Upload</button>
    </form>
  </body>
</html>
"""

SELECT_POINTS_HTML = """<!doctype html>
<html>
  <body>
    <h3>Select Start then Goal</h3>
    <p>Click once for Start (magenta), then once for Goal (yellow). Coordinates shown below.</p>
    <canvas id="cnv"></canvas>
    <div id="coords">Start: -, Goal: -</div>
    <form id="goForm" method="post" action="/start">
      <input type="hidden" name="jobId" value="{jobId}">
      <input type="hidden" name="startY" id="startY">
      <input type="hidden" name="startX" id="startX">
      <input type="hidden" name="goalY" id="goalY">
      <input type="hidden" name="goalX" id="goalX">
      <label>Expansions per frame: <input type="number" name="expansionsPerFrame" value="25" min="1" max="200"></label>
      <button type="submit">Run</button>
    </form>
    <script>
      const jobId = "{jobId}";
      const imgUrl = "/image/" + jobId + "/base.png";
      const cnv = document.getElementById("cnv");
      const ctx = cnv.getContext("2d");
      const coordsDiv = document.getElementById("coords");
      const startY = document.getElementById("startY");
      const startX = document.getElementById("startX");
      const goalY = document.getElementById("goalY");
      const goalX = document.getElementById("goalX");
      let clicks = [];
      const img = new Image();
      img.onload = () => {{ cnv.width = img.width; cnv.height = img.height; ctx.drawImage(img,0,0); }};
      img.src = imgUrl;

      function drawMarkers() {{
        ctx.drawImage(img,0,0);
        if (clicks[0]) {{
          ctx.fillStyle = "magenta";
          ctx.fillRect(clicks[0].x-2, clicks[0].y-2, 5, 5);
        }}
        if (clicks[1]) {{
          ctx.fillStyle = "yellow";
          ctx.fillRect(clicks[1].x-2, clicks[1].y-2, 5, 5);
        }}
      }}
      cnv.addEventListener("click", (e) => {{
        const rect = cnv.getBoundingClientRect();
        const x = Math.round(e.clientX - rect.left);
        const y = Math.round(e.clientY - rect.top);
        if (clicks.length < 2) {{ clicks.push({{x, y}}); }} else {{ clicks = [{{x, y}}]; }}
        drawMarkers();
        if (clicks[0]) {{ startY.value = clicks[0].y; startX.value = clicks[0].x; }}
        if (clicks[1]) {{ goalY.value = clicks[1].y; goalX.value = clicks[1].x; }}
        coordsDiv.textContent = "Start: " + (clicks[0] ? clicks[0].y + "," + clicks[0].x : "-") +
                                " | Goal: " + (clicks[1] ? clicks[1].y + "," + clicks[1].x : "-");
      }});
    </script>
  </body>
</html>
"""

RUN_HTML = """<!doctype html>
<html>
  <body>
    <h3>Processing</h3>
    <p>This page streams logs live. Please wait.</p>
    <img id="frame" src="/image/{jobId}/current.png?ts={ts}" alt="Current frame" width="700"/>
    <pre id="log" style="white-space: pre-wrap;"></pre>

    <script>
      const jobId = "{jobId}";
      const frameEl = document.getElementById("frame");
      const logEl = document.getElementById("log");
      const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/" + jobId);

      function refreshFrame() {{
        frameEl.src = "/image/" + jobId + "/current.png?ts=" + Date.now();
      }}

      ws.onmessage = (ev) => {{
        try {{
          const msg = JSON.parse(ev.data);
          if (msg.type === "log") {{
            logEl.textContent += msg.msg + "\\n";
          }} else if (msg.type === "frame") {{
            refreshFrame();
          }} else if (msg.type === "done") {{
            logEl.textContent += "[done]\\n";
            ws.close();
            // Redirect after 1.5s
            setTimeout(() => {{
              window.location.href = "/result/" + jobId;
            }}, 1500);
          }}
        }} catch(e) {{
          logEl.textContent += "[parse-error] " + e + "\\n";
        }}
      }};

      ws.onopen = () => {{ logEl.textContent += "Connected. Streaming logs...\\n"; }};
      ws.onerror = () => {{ logEl.textContent += "WebSocket error.\\n"; }};
      ws.onclose = () => {{ logEl.textContent += "Closed.\\n"; }};

      setInterval(refreshFrame, 1000);
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.post("/upload", response_class=HTMLResponse)
async def upload(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file")
    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported format")
    uploadsDir = Path("uploads")
    uploadsDir.mkdir(parents=True, exist_ok=True)
    dst = uploadsDir / f"{uuid.uuid4().hex}{suffix}"
    data = await image.read()
    dst.write_bytes(data)

    job = JOB_STORE.create(dst)
    # prepare base image for selection page
    try:
        rgb = loadImage(str(dst), maxDim=900)
        job.rgb = rgb
        job.originalRgb = rgb.copy()
        job.currentFrame = rgb.copy()
        basePng = ndarrayToPngBytes(rgb)
        baseDir = job.outputRoot / "image"
        baseDir.mkdir(parents=True, exist_ok=True)
        (baseDir / "base.png").write_bytes(basePng)
    except Exception as exc:
        job.error = str(exc)
        return PlainTextResponse(f"Image load failed: {exc}", status_code=500)

    html = SELECT_POINTS_HTML.format(jobId=job.jobId)
    return HTMLResponse(html)

@app.get("/image/{jobId}/base.png")
def get_base(jobId: str):
    job = JOB_STORE.get(jobId)
    if job.rgb is None:
        raise HTTPException(status_code=404, detail="Base not ready")
    png = ndarrayToPngBytes(job.rgb)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")

@app.get("/image/{jobId}/current.png")
def get_current(jobId: str):
    job = JOB_STORE.get(jobId)
    arr = job.currentFrame if job.currentFrame is not None else job.rgb
    if arr is None:
        raise HTTPException(status_code=404, detail="No frame")
    png = ndarrayToPngBytes(arr)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")

@app.post("/start", response_class=HTMLResponse)
def start(
    backgroundTasks: BackgroundTasks,
    jobId: str = Form(...),
    startY: int = Form(...),
    startX: int = Form(...),
    goalY: int = Form(...),
    goalX: int = Form(...),
    expansionsPerFrame: int = Form(25),
):
    job = JOB_STORE.get(jobId)
    if job.rgb is None:
        raise HTTPException(status_code=400, detail="Image not loaded")
    # Clamp to bounds
    h, w = job.rgb.shape[0], job.rgb.shape[1]
    sY = max(0, min(int(startY), h - 1))
    sX = max(0, min(int(startX), w - 1))
    gY = max(0, min(int(goalY), h - 1))
    gX = max(0, min(int(goalX), w - 1))
    job.start = (sY, sX)
    job.goal = (gY, gX)
    job.log(f"Start set to {job.start}")
    job.log(f"Goal set to {job.goal}")
    job.log(f"Expansions per frame: {expansionsPerFrame}")

    # Kick background processing
    backgroundTasks.add_task(runPipeline, job, expansionsPerFrame, 12)
    return HTMLResponse(RUN_HTML.format(jobId=jobId, ts=int(time.time()*1000)))

@app.get("/status/{jobId}")
def status(jobId: str):
    job = JOB_STORE.get(jobId)
    return {
        "done": job.done,
        "error": job.error,
        "summaryPath": f"/summary/{jobId}.png" if job.summaryImagePath and job.summaryImagePath.exists() else None,
        "videoPath": f"/video/{jobId}.mp4" if job.finalVideoPath and job.finalVideoPath.exists() else None,
    }

@app.get("/summary/{jobId}.png")
def get_summary(jobId: str):
    job = JOB_STORE.get(jobId)
    if not job.summaryImagePath or not job.summaryImagePath.exists():
        raise HTTPException(status_code=404, detail="Summary not available")
    return FileResponse(job.summaryImagePath, media_type="image/png")

@app.get("/video/{jobId}.mp4")
def get_video(jobId: str):
    job = JOB_STORE.get(jobId)
    if not job.finalVideoPath or not job.finalVideoPath.exists():
        raise HTTPException(status_code=404, detail="Video not available")
    return FileResponse(job.finalVideoPath, media_type="video/mp4", filename="search_animation.mp4")

# ---------------- WebSocket Log Stream ----------------
@app.websocket("/ws/{jobId}")
async def ws_logs(ws: WebSocket, jobId: str):
    await ws.accept()
    try:
        job = JOB_STORE.get(jobId)
    except KeyError:
        await ws.send_text(json.dumps({"type": "log", "msg": "Job not found"}))
        await ws.close()
        return

    lastLen = 0
    try:
        # Push existing logs
        for msg in list(job.logs):
            await ws.send_text(json.dumps({"type": "log", "msg": msg}))

        # Stream updates
        while True:
            await asyncio.sleep(0.3)
            # Send new logs
            with job.lock:
                if len(job.logs) > lastLen:
                    for idx in range(lastLen, len(job.logs)):
                        await ws.send_text(json.dumps({"type": "log", "msg": job.logs[idx]}))
                    lastLen = len(job.logs)
            # Send frame tick
            await ws.send_text(json.dumps({"type": "frame"}))
            if job.done:
                await ws.send_text(json.dumps({"type": "done"}))
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "log", "msg": f'WebSocket error: {exc}'}))
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass
@app.get("/result/{jobId}", response_class=HTMLResponse)
def result_page(jobId: str):
    job = JOB_STORE.get(jobId)
    if not job.done:
        return HTMLResponse(f"<h3>Job {jobId} still running...</h3>", status_code=202)

    html = f"""<!doctype html>
<html>
  <body>
    <h2>Processing Complete</h2>
    <h3>Summary Image</h3>
    <img src="/summary/{jobId}.png" width="700" alt="summary image" />

    <h3>Search Animation</h3>
    <video width="700" controls autoplay loop>
      <source src="/video/{jobId}.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

    <br><br>
    <a href="/">← Upload Another Image</a>
  </body>
</html>
"""
    return HTMLResponse(html)

@app.get("/stream/video/{jobId}")
def stream_video(jobId: str):
    job = JOB_STORE.get(jobId)
    if not job.finalVideoPath or not job.finalVideoPath.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Serve video properly for browsers (supports seeking)
    return FileResponse(
        job.finalVideoPath,
        media_type="video/mp4",
        filename=f"{jobId}.mp4",
        headers={"Accept-Ranges": "bytes"}
    )
