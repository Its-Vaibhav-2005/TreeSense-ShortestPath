import heapq
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from PIL import Image


Point = Tuple[int, int]


# ---------- I/O ----------
def load_image(path: str, max_dim: int = 900) -> np.ndarray:
    """Load an RGB image, optionally resizing it while keeping aspect ratio."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    scale = min(max_dim / width, max_dim / height, 1.0)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        try:
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:
            img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)


# ---------- Vegetation mask and costs ----------
def compute_tree_mask(rgb: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Return a binary mask where 1 approximates tree/vegetation pixels."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    max_rb = np.maximum(r, b)
    exg = (g - max_rb) / (g + max_rb + 1e-6)
    return (exg > threshold).astype(np.int32)


def build_cost_map(tree_mask: np.ndarray, tree_cost: int = 20, open_cost: int = 1) -> np.ndarray:
    """Assign traversal cost to each pixel: heavier inside vegetation."""
    return np.where(tree_mask == 1, tree_cost, open_cost).astype(np.int32)


# ---------- Point picking ----------
def pick_points(img: np.ndarray) -> Tuple[Point, Point]:
    fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.set_title("Click START then GOAL. Close window if stuck.")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Exactly two points required.")

    def clamp(pt):
        row = int(round(pt[1]))
        col = int(round(pt[0]))
        row = min(max(0, row), img.shape[0] - 1)
        col = min(max(0, col), img.shape[1] - 1)
        return row, col

    return clamp(pts[0]), clamp(pts[1])


# ---------- A* driven by animation ----------
class AStarAnimator:
    def __init__(
        self,
        rgb: np.ndarray,
        cost_map: np.ndarray,
        start: Point,
        goal: Point,
        expansions_per_frame: int = 10,
    ) -> None:
        self.rgb = rgb
        self.cost = cost_map
        self.start = start
        self.goal = goal
        self.expansions_per_frame = max(1, expansions_per_frame)

        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.g = {start: 0}
        self.f = {start: self._heuristic(start, goal)}
        self.parents: dict[Point, Point] = {}
        self.open: List[Tuple[float, Point]] = [(self.f[start], start)]
        self.closed: set[Point] = set()
        self.done = False
        self.path: List[Point] = []
        self.last_current: Point | None = None

        # Base frame emphasising vegetation for easier viewing
        self.base = rgb.copy()
        mask = cost_map > cost_map.min()
        green_tint = np.zeros_like(self.base)
        green_tint[:] = [0, 140, 0]
        self.base[mask] = (
            0.4 * self.base[mask].astype(np.float32)
            + 0.6 * green_tint[mask].astype(np.float32)
        ).astype(np.uint8)

        self.frame_buffer = np.empty_like(self.base)
        self.visited_mask = np.zeros(cost_map.shape, dtype=bool)
        self.frontier_mask = np.zeros(cost_map.shape, dtype=bool)

    @staticmethod
    def _heuristic(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _advance(self) -> None:
        for _ in range(self.expansions_per_frame):
            if self.done:
                return
            if not self.open:
                self.done = True
                return

            _, current = heapq.heappop(self.open)
            if current in self.closed:
                continue

            self.closed.add(current)
            self.visited_mask[current] = True
            self.last_current = current

            if current == self.goal:
                path: List[Point] = [current]
                while current in self.parents:
                    current = self.parents[current]
                    path.append(current)
                self.path = list(reversed(path))
                self.done = True
                return

            current_cost = self.g[current]
            for dy, dx in self.neighbors:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < self.cost.shape[0] and 0 <= nx < self.cost.shape[1]:
                    tentative = current_cost + int(self.cost[ny, nx])
                    neighbor = (ny, nx)
                    if tentative < self.g.get(neighbor, float("inf")):
                        self.parents[neighbor] = current
                        self.g[neighbor] = tentative
                        self.f[neighbor] = tentative + self._heuristic(neighbor, self.goal)
                        heapq.heappush(self.open, (self.f[neighbor], neighbor))

    def step(self) -> np.ndarray:
        self._advance()

        np.copyto(self.frame_buffer, self.base)

        if self.visited_mask.any():
            self.frame_buffer[self.visited_mask] = [100, 149, 237]

        self.frontier_mask.fill(False)
        for _, node in self.open:
            if node not in self.closed:
                self.frontier_mask[node] = True
        if self.frontier_mask.any():
            self.frame_buffer[self.frontier_mask] = [255, 165, 0]

        if self.last_current is not None and not self.done:
            cy, cx = self.last_current
            self.frame_buffer[cy, cx] = [65, 105, 225]

        sy, sx = self.start
        gy, gx = self.goal
        self.frame_buffer[sy, sx] = [255, 0, 255]
        self.frame_buffer[gy, gx] = [255, 255, 0]

        if self.path:
            for py, px in self.path:
                self.frame_buffer[py, px] = [255, 0, 0]

        return self.frame_buffer


# ---------- Optional interactive preview ----------
def animate_search(anim: AStarAnimator, interval_ms: int = 20):
    fig, ax = plt.subplots(figsize=(8, 8 * anim.rgb.shape[0] / anim.rgb.shape[1]))
    ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
    ax.axis("off")
    im = ax.imshow(anim.step())

    def update(_):
        im.set_data(anim.step())
        return [im]

    animation = FuncAnimation(fig, update, interval=interval_ms, blit=False, repeat=False)
    plt.show()
    return animation


def plot_paths_summary(
    anim: AStarAnimator,
    rgb: np.ndarray,
    original: np.ndarray,
    cost_map: np.ndarray,
):
    if not anim.path:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rows = [pt[0] for pt in anim.path]
    cols = [pt[1] for pt in anim.path]

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    cost_vis = cost_map.astype(np.float32)
    axes[0, 1].imshow(cost_vis, cmap="viridis")
    axes[0, 1].set_title("Cost Map (Higher = Trees)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb)
    axes[1, 0].plot(cols, rows, color="red", linewidth=3, alpha=0.9)
    axes[1, 0].set_title("Best Path Overlay")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylim(rgb.shape[0], 0)

    axes[1, 1].imshow(rgb)
    explored_y, explored_x = np.where(anim.visited_mask)
    if explored_y.size:
        axes[1, 1].scatter(explored_x, explored_y, s=2, c="gold", alpha=0.3, label="Explored")
    axes[1, 1].plot(cols, rows, color="red", linewidth=3, alpha=0.9, label="Best Path")
    axes[1, 1].set_title("Explored vs Best Path")
    axes[1, 1].axis("off")
    axes[1, 1].set_ylim(rgb.shape[0], 0)
    axes[1, 1].legend(loc="upper right")

    fig.subplots_adjust(
        hspace=0.095,
        wspace=0.062,
        top=0.945,
        bottom=0.015,
        left=0.010,
        right=0.990,
    )

    return fig


def fallback_straight_path(start: Point, goal: Point) -> List[Point]:
    """Generate a straight-line path using Bresenham's algorithm."""
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


def save_animation(
    anim: AStarAnimator,
    interval_ms: int,
    output_path: Path,
    hold_frames: int = 15,
    dpi: int = 120,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1000 / interval_ms)))

    ffmpeg_path = plt.rcParams.get("animation.ffmpeg_path")
    resolved_ffmpeg: str | None = None
    if ffmpeg_path and Path(ffmpeg_path).is_file():
        resolved_ffmpeg = ffmpeg_path
    else:
        found = shutil.which("ffmpeg")
        if found:
            resolved_ffmpeg = found
        else:
            try:
                import imageio_ffmpeg  # type: ignore[import-not-found]

                resolved_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                raise RuntimeError(
                    "ffmpeg executable not found. Install ffmpeg (e.g. `conda install -c conda-forge ffmpeg`), "
                    "or install the `imageio-ffmpeg` package."
                ) from exc

    plt.rcParams["animation.ffmpeg_path"] = resolved_ffmpeg

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)

    fig, ax = plt.subplots(figsize=(8, 8 * anim.rgb.shape[0] / anim.rgb.shape[1]))
    ax.set_title("A* Pathfinding — Blue: explored, Red: best path, Green: vegetation")
    ax.axis("off")

    frame = anim.step()
    im = ax.imshow(frame)

    try:
        with writer.saving(fig, str(output_path), dpi):
            writer.grab_frame()
            while not anim.done:
                frame = anim.step()
                im.set_data(frame)
                writer.grab_frame()
            for _ in range(hold_frames):
                writer.grab_frame()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save animation to {output_path}. Ensure ffmpeg is installed and reachable."
        ) from exc
    finally:
        plt.close(fig)

    return output_path


# ---------- Main ----------
if __name__ == "__main__":
    image_path = "./jungleShot.jpg"
    rgb = load_image(image_path, max_dim=900)
    original_rgb = rgb.copy()

    tree_mask = compute_tree_mask(rgb, threshold=0.15)
    cost_map = build_cost_map(tree_mask, tree_cost=20, open_cost=1)

    start, goal = pick_points(rgb)
    print(f"Start: {start}, Goal: {goal}")

    output_root = Path("BestPath-JungleMode")
    video_dir = output_root / "video"
    image_dir = output_root / "image"

    animator = AStarAnimator(rgb, cost_map, start, goal, expansions_per_frame=25)
    video_path = video_dir / "search_animation.mp4"

    final_video_path: Path | None = None
    try:
        print(f"Saving search animation to {video_path} ...")
        final_video_path = save_animation(animator, interval_ms=12, output_path=video_path)
    except RuntimeError as err:
        print(err)
        print("Continuing without saving the animation video.")
        while not animator.done:
            animator.step()

    if animator.path:
        trees_crossed = int(np.sum(tree_mask[tuple(np.array(animator.path).T)]))
        print(f"Path length: {len(animator.path)}")
        print(f"Estimated trees crossed: {trees_crossed}")

        summary_fig = plot_paths_summary(animator, animator.base, original_rgb, cost_map)
        if summary_fig is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            summary_path = image_dir / "best_path_summary.png"
            summary_fig.savefig(summary_path, bbox_inches="tight", dpi=160)
            plt.close(summary_fig)
            print(f"Summary figure saved to {summary_path}")

        if final_video_path is not None:
            print(f"Search animation saved to {final_video_path}")
    else:
        print("No viable path found with current costs. Falling back to straight traversal.")
        direct_path = fallback_straight_path(start, goal)
        trees_crossed = int(np.sum(tree_mask[tuple(np.array(direct_path).T)]))
        print(f"Direct path length: {len(direct_path)}")
        print(f"Estimated trees encountered on direct path: {trees_crossed}")
        if final_video_path is not None:
            print(f"Search animation saved to {final_video_path}")
