from __future__ import annotations

import asyncio
import heapq
import io
import json
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import imageio_ffmpeg
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
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image

# ---------- Types ----------
Point = Tuple[int, int]

# ---------- Paths / Templates ----------
ROOT_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = ROOT_DIR / "uploads"
OUTPUT_ROOT = ROOT_DIR
TEMPLATES = Jinja2Templates(directory=str(ROOT_DIR / "templates"))

# ---------- I/O Helpers ----------
def LoadImage(path: str, maxDim: int = 900) -> np.ndarray:
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

def ComputeTreeMask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    maxRb = np.maximum(r, b)
    exg = (g - maxRb) / (g + maxRb + 1e-6)
    return (exg > threshold).astype(np.int32)

def BuildCostMap(treeMask: np.ndarray, treeCost: int = 20, openCost: int = 1) -> np.ndarray:
    return np.where(treeMask == 1, treeCost, openCost).astype(np.int32)

def NdarrayToPngBytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# ---------- A* Animator ----------
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
        self.f = {start: self._Heuristic(start, goal)}
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
    def _Heuristic(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _Advance(self) -> None:
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
                        self.f[neighbor] = tentative + self._Heuristic(neighbor, self.goal)
                        heapq.heappush(self.open, (self.f[neighbor], neighbor))

    def Step(self) -> np.ndarray:
        self._Advance()
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

# ---------- Summary Plot ----------
def PlotPathsSummary(anim: AStarAnimator, rgb: np.ndarray, original: np.ndarray, costMap: np.ndarray):
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

def FallbackStraightPath(start: Point, goal: Point) -> List[Point]:
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

# ---------- Video Save ----------
def SaveAnimation(
    anim: AStarAnimator,
    intervalMs: int,
    outputPath: Path,
    holdFrames: int = 15,
) -> Path:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    ffmpegPath = imageio_ffmpeg.get_ffmpeg_exe()
    fps = max(1, int(round(1000 / intervalMs)))
    h, w = anim.rgb.shape[:2]
    size = f"{w}x{h}"

    process = subprocess.Popen(
        [
            ffmpegPath,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", size,
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(outputPath),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        frame = anim.Step()
        process.stdin.write(frame.tobytes())

        while not anim.done:
            frame = anim.Step()
            process.stdin.write(frame.tobytes())

        for _ in range(holdFrames):
            process.stdin.write(anim.frameBuffer.tobytes())

        process.stdin.close()
        stderrOutput = process.stderr.read().decode("utf-8", errors="ignore")
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (code {process.returncode}): {stderrOutput}")

        if not outputPath.exists() or outputPath.stat().st_size == 0:
            raise RuntimeError("ffmpeg wrote no output or empty file")

    except Exception as exc:
        try:
            process.kill()
        except Exception:
            pass
        raise RuntimeError(f"Failed to save animation video: {exc}") from exc
    finally:
        try:
            if process.poll() is None:
                process.terminate()
            process.stderr.close()
        except Exception:
            pass

    return outputPath

# ---------- Job Model ----------
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

    def Log(self, msg: str) -> None:
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {msg}")

# ---------- Job Store ----------
class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def Create(self, uploadPath: Path) -> Job:
        jobId = uuid.uuid4().hex[:12]
        outRoot = OUTPUT_ROOT / "BestPath-JungleMode" / jobId
        job = Job(jobId=jobId, uploadPath=uploadPath, outputRoot=outRoot)
        with self._lock:
            self._jobs[jobId] = job
        return job

    def Get(self, jobId: str) -> Job:
        with self._lock:
            if jobId not in self._jobs:
                raise KeyError("Job not found")
            return self._jobs[jobId]

JOB_STORE = JobStore()

# ---------- Processing ----------
def RunPipeline(job: Job, expansionsPerFrame: int = 25, intervalMs: int = 12) -> None:
    try:
        job.Log("Loading image")
        rgb = LoadImage(str(job.uploadPath), maxDim=900)
        job.rgb = rgb
        job.originalRgb = rgb.copy()
        job.currentFrame = rgb.copy()

        job.Log("Computing vegetation mask")
        treeMask = ComputeTreeMask(rgb, threshold=0.15)
        job.treeMask = treeMask

        job.Log("Building traversal cost map")
        costMap = BuildCostMap(treeMask, treeCost=20, openCost=1)
        job.costMap = costMap

        if job.start is None or job.goal is None:
            job.error = "Start and Goal not set"
            job.Log("Error: start and goal not set")
            job.done = True
            return

        job.Log(f"Starting A* search from {job.start} to {job.goal}")
        animator = AStarAnimator(rgb, costMap, job.start, job.goal, expansionsPerFrame=expansionsPerFrame)
        job.animator = animator

        videoDir = job.outputRoot / "video"
        imageDir = job.outputRoot / "image"
        videoDir.mkdir(parents=True, exist_ok=True)
        imageDir.mkdir(parents=True, exist_ok=True)
        videoPath = videoDir / "search_animation.mp4"

        try:
            job.Log("Writing video with imageio-ffmpeg")
            finalPath = SaveAnimation(animator, intervalMs=intervalMs, outputPath=videoPath, holdFrames=15)
            job.finalVideoPath = finalPath
            job.Log(f"Video saved: {finalPath}")
        except Exception as exc:
            job.Log(f"Video export failed: {exc}")
            while not animator.done:
                frame = animator.Step()
                job.currentFrame = frame.copy()

        if animator.done:
            job.currentFrame = animator.frameBuffer.copy()

        if animator.path:
            treesCrossed = int(np.sum(treeMask[tuple(np.array(animator.path).T)]))
            job.Log(f"Path length: {len(animator.path)}")
            job.Log(f"Estimated trees crossed: {treesCrossed}")

            job.Log("Creating summary figure")
            fig2 = PlotPathsSummary(animator, animator.base, job.originalRgb, costMap)
            if fig2 is not None:
                summaryPath = job.outputRoot / "image" / "best_path_summary.png"
                fig2.savefig(summaryPath, bbox_inches="tight", dpi=160)
                plt.close(fig2)
                job.summaryImagePath = summaryPath
                job.Log(f"Summary saved: {summaryPath}")
        else:
            job.Log("No viable path found. Generating straight-line fallback")
            direct = FallbackStraightPath(job.start, job.goal)
            treesCrossed = int(np.sum(treeMask[tuple(np.array(direct).T)]))
            job.Log(f"Direct path length: {len(direct)}")
            job.Log(f"Estimated trees encountered on direct path: {treesCrossed}")

        job.done = True
        job.Log("Processing complete")
    except Exception as exc:
        job.error = str(exc)
        job.done = True
        job.Log(f"Fatal error: {exc}")

# ---------- FastAPI App ----------
app = FastAPI(title="Jungle Path A* — Minimal UI")

# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def Index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def Upload(request: Request, image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file")
    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported format")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    data = await image.read()
    dst.write_bytes(data)

    job = JOB_STORE.Create(dst)
    try:
        rgb = LoadImage(str(dst), maxDim=900)
        job.rgb = rgb
        job.originalRgb = rgb.copy()
        job.currentFrame = rgb.copy()
        basePng = NdarrayToPngBytes(rgb)
        baseDir = job.outputRoot / "image"
        baseDir.mkdir(parents=True, exist_ok=True)
        (baseDir / "base.png").write_bytes(basePng)
    except Exception as exc:
        job.error = str(exc)
        return PlainTextResponse(f"Image load failed: {exc}", status_code=500)

    return TEMPLATES.TemplateResponse("selectPoint.html", {"request": request, "jobId": job.jobId})

@app.get("/image/{jobId}/base.png")
def GetBase(jobId: str):
    job = JOB_STORE.Get(jobId)
    if job.rgb is None:
        raise HTTPException(status_code=404, detail="Base not ready")
    png = NdarrayToPngBytes(job.rgb)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")

@app.get("/image/{jobId}/current.png")
def GetCurrent(jobId: str):
    job = JOB_STORE.Get(jobId)
    arr = job.currentFrame if job.currentFrame is not None else job.rgb
    if arr is None:
        raise HTTPException(status_code=404, detail="No frame")
    png = NdarrayToPngBytes(arr)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")

@app.post("/start", response_class=HTMLResponse)
def Start(
    request: Request,
    backgroundTasks: BackgroundTasks,
    jobId: str = Form(...),
    startY: int = Form(...),
    startX: int = Form(...),
    goalY: int = Form(...),
    goalX: int = Form(...),
    expansionsPerFrame: int = Form(25),
):
    job = JOB_STORE.Get(jobId)
    if job.rgb is None:
        raise HTTPException(status_code=400, detail="Image not loaded")
    h, w = job.rgb.shape[0], job.rgb.shape[1]
    sY = max(0, min(int(startY), h - 1))
    sX = max(0, min(int(startX), w - 1))
    gY = max(0, min(int(goalY), h - 1))
    gX = max(0, min(int(goalX), w - 1))
    job.start = (sY, sX)
    job.goal = (gY, gX)
    job.Log(f"Start set to {job.start}")
    job.Log(f"Goal set to {job.goal}")
    job.Log(f"Expansions per frame: {expansionsPerFrame}")

    backgroundTasks.add_task(RunPipeline, job, expansionsPerFrame, 12)
    return TEMPLATES.TemplateResponse(
        "run.html",
        {
            "request": request,
            "jobId": jobId,
            "ts": int(time.time() * 1000),
        },
    )

@app.get("/status/{jobId}")
def Status(jobId: str):
    job = JOB_STORE.Get(jobId)
    return {
        "done": job.done,
        "error": job.error,
        "summaryPath": f"/summary/{jobId}.png" if job.summaryImagePath and job.summaryImagePath.exists() else None,
        "videoPath": f"/stream/video/{jobId}" if job.finalVideoPath and job.finalVideoPath.exists() else None,
    }

@app.get("/summary/{jobId}.png")
def GetSummary(jobId: str):
    job = JOB_STORE.Get(jobId)
    if not job.summaryImagePath or not job.summaryImagePath.exists():
        raise HTTPException(status_code=404, detail="Summary not available")
    return FileResponse(job.summaryImagePath, media_type="image/png")

@app.get("/video/{jobId}.mp4")
def GetVideoLegacy(jobId: str):
    job = JOB_STORE.Get(jobId)
    if not job.finalVideoPath or not job.finalVideoPath.exists():
        raise HTTPException(status_code=404, detail="Video not available")
    return FileResponse(job.finalVideoPath, media_type="video/mp4", filename="search_animation.mp4")

@app.get("/stream/video/{jobId}")
def StreamVideo(jobId: str):
    job = JOB_STORE.Get(jobId)
    if not job.finalVideoPath or not job.finalVideoPath.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        job.finalVideoPath,
        media_type="video/mp4",
        filename=f"{jobId}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )

# ---------- WebSocket ----------
@app.websocket("/ws/{jobId}")

async def WsLogs(ws: WebSocket, jobId: str):
    await ws.accept()
    try:
        job = JOB_STORE.Get(jobId)
    except KeyError:
        await ws.send_text(json.dumps({"type": "log", "msg": "Job not found"}))
        await ws.close()
        return

    lastLen = 0
    try:
        for msg in list(job.logs):
            await ws.send_text(json.dumps({"type": "log", "msg": msg}))

        while True:
            await asyncio.sleep(0.3)
            with job.lock:
                if len(job.logs) > lastLen:
                    for idx in range(lastLen, len(job.logs)):
                        await ws.send_text(json.dumps({"type": "log", "msg": job.logs[idx]}))
                    lastLen = len(job.logs)
            await ws.send_text(json.dumps({"type": "frame"}))
            if job.done:
                await ws.send_text(json.dumps({"type": "done"}))
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "log", "msg": f"WebSocket error: {exc}"}))
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

# ---------- Result Page ----------
@app.get("/result/{jobId}", response_class=HTMLResponse)
def ResultPage(request: Request, jobId: str):
    job = JOB_STORE.Get(jobId)
    if not job.done:
        return HTMLResponse(f"<h3>Job {jobId} still running...</h3>", status_code=202)

    summaryAvailable = job.summaryImagePath and job.summaryImagePath.exists()
    videoAvailable = job.finalVideoPath and job.finalVideoPath.exists()

    return TEMPLATES.TemplateResponse(
        "result.html",
        {
            "request": request,
            "jobId": jobId,
            "summaryAvailable": bool(summaryAvailable),
            "videoAvailable": bool(videoAvailable),
        },
    )
