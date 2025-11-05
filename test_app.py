from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import json

from main import app

client = TestClient(app)


def create_white_png(w=64, h=48):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


def main():
    print("Starting automated test using TestClient...")

    # 1) Upload
    img_bytes = create_white_png()
    files = {"file": ("test.png", img_bytes, "image/png")}
    resp = client.post("/upload", files=files, allow_redirects=False)
    print("Upload response status:", resp.status_code)
    if resp.status_code not in (303, 200):
        print("Unexpected upload response:", resp.text)
        return 1

    # Extract image_id from redirect location
    location = resp.headers.get("location")
    if not location:
        print("No Location header found after upload; response headers:", resp.headers)
        return 1
    print("Redirected to:", location)
    # expected /select/{image_id}
    image_id = location.rstrip("/").split("/")[-1]
    print("Detected image_id:", image_id)

    # 2) Solve with JSON API
    payload = {
        "image_id": image_id,
        "start": [1, 1],
        "goal": [62, 46],
        "waypoints": [],
        "obstacles": [],
        "dark_threshold": 80,
        "neighbor_mode": "4",
        "heuristic": "manhattan",
    }
    resp2 = client.post("/solve", json=payload)
    print("Solve response status:", resp2.status_code)
    if resp2.status_code != 200:
        print("Solve failed:", resp2.text)
        return 1

    data = resp2.json()
    print("Solve response keys:", list(data.keys()))

    # 3) Fetch summary and animation
    summary_url = data.get("summary_url")
    animation_url = data.get("animation_url")
    animation_media_type = data.get("animation_media_type")
    print("Summary URL:", summary_url)
    print("Animation URL:", animation_url)
    print("Animation media type:", animation_media_type)

    if summary_url:
        rsum = client.get(summary_url)
        print("Summary GET status:", rsum.status_code)
        if rsum.status_code == 200:
            print("Summary length bytes:", len(rsum.content))
        else:
            print("Failed to get summary:", rsum.text)

    if animation_url:
        rvid = client.get(animation_url)
        print("Animation GET status:", rvid.status_code)
        if rvid.status_code == 200:
            print("Animation length bytes:", len(rvid.content))
            print("Animation content-type header:", rvid.headers.get("content-type"))
        else:
            print("Failed to get animation (this may be due to missing ffmpeg in the environment):", rvid.text)

    # Print a small path sample
    path = data.get("path")
    if path:
        print("Returned path length:", len(path))
        print("First 8 path points:", path[:8])

    print("Automated test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
