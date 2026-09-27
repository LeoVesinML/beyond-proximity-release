#!/usr/bin/env python3
"""Record a live end-to-end product demo MP4 (browser + real UI).

Automates SemanticSplat in Chromium: load 3D scene, capture view, run query,
watch Query Flow pipeline, show found location.

Usage:
    PYTHONPATH=. python scripts/record_e2e_product_demo.py
    PYTHONPATH=. python scripts/record_e2e_product_demo.py --query "Find the sofa"

Requires: playwright (`pip install playwright && playwright install chromium`), ffmpeg.

Output: docs/benchmark_results/e2e_product_demo.mp4
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "benchmark_results" / "e2e_product_demo.mp4"
VIDEO_TMP = ROOT / "docs" / "benchmark_results" / "_e2e_recording"
FRONTEND_URL = "http://localhost:5173"
HEALTH_URL = "http://127.0.0.1:8000/api/health"
DEFAULT_QUERY = "Find the sofa"


def _ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return
    except ImportError:
        print("Installing playwright…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _start_servers() -> List[subprocess.Popen]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    procs: List[subprocess.Popen] = []
    if not _http_ok(HEALTH_URL):
        print("Starting backend…")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "backend.api.server"],
            cwd=ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    if not _http_ok(FRONTEND_URL):
        print("Starting frontend…")
        procs.append(subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ROOT / "frontend",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    for _ in range(60):
        if _http_ok(HEALTH_URL) and _http_ok(FRONTEND_URL):
            print("Services ready.")
            return procs
        time.sleep(1)
    raise RuntimeError("Services did not start in 60s — run ./start_all.sh first")


def _stop_servers(procs: List[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=5)
        except Exception:
            p.kill()


def _webm_to_mp4(webm: Path, mp4: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(webm),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def record_demo(query: str, output: Path, viewport: tuple[int, int] = (1280, 720)) -> Path:
    _ensure_playwright()
    from playwright.sync_api import sync_playwright

    spawned = _start_servers()
    if VIDEO_TMP.exists():
        shutil.rmtree(VIDEO_TMP)
    VIDEO_TMP.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
                record_video_dir=str(VIDEO_TMP),
                record_video_size={"width": viewport[0], "height": viewport[1]},
            )
            page = context.new_page()

            print("Opening app…")
            page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(3000)

            # Scene init
            print("Setting scene default…")
            scene_input = page.locator('input[placeholder="default"]')
            if scene_input.count():
                scene_input.fill("default")
            page.get_by_role("button", name="Set").click()
            page.wait_for_timeout(800)

            # Load PLY
            print("Loading ConferenceHall.ply…")
            select = page.locator("select").first
            if select.count():
                select.select_option(label="ConferenceHall.ply")
            else:
                ply_input = page.locator('input[placeholder*="ConferenceHall"]')
                if ply_input.count():
                    ply_input.fill("/scenes/ConferenceHall.ply")
                    page.get_by_role("button", name="Load").click()
            page.wait_for_timeout(5000)

            # Navigator tab is default — fly + capture
            print("Navigator — capture view (R)…")
            canvas = page.locator("canvas").first
            canvas.wait_for(state="visible", timeout=30_000)
            canvas.click()
            for key in ("w", "w", "d", "d"):
                page.keyboard.press(key)
                page.wait_for_timeout(350)
            page.keyboard.press("r")
            page.wait_for_timeout(2000)

            # Query
            print(f"Running query: {query!r}…")
            query_input = page.locator('input[placeholder*="sofa"], input[placeholder*="Find"]').first
            query_input.fill(query)
            page.get_by_role("button", name="Ask").click()

            # Query Flow tab (auto-opens after Ask)
            page.wait_for_timeout(1000)
            page.locator("header button").filter(has_text=re.compile("Query Flow", re.I)).click()

            # Wait for pipeline completion (up to 30s)
            print("Waiting for pipeline…")
            for _ in range(30):
                page.wait_for_timeout(1000)
                if page.locator("text=/found/i").count():
                    break
                if page.locator("text=/not.?found/i").count():
                    break

            page.wait_for_timeout(2000)
            page.mouse.wheel(0, 500)
            page.wait_for_timeout(1500)
            page.mouse.wheel(0, 500)
            page.wait_for_timeout(2000)

            page.locator("header button").filter(has_text=re.compile("Semantic Tree", re.I)).click()
            page.wait_for_timeout(2500)
            page.wait_for_timeout(1000)

            page.close()
            video_path = page.video.path() if page.video else None
            context.close()
            browser.close()

        if not video_path or not video_path.exists():
            webms = list(VIDEO_TMP.glob("*.webm"))
            if not webms:
                raise RuntimeError("No video recorded")
            video_path = webms[0]

        print(f"Converting {video_path.name} → MP4…")
        _webm_to_mp4(video_path, output)
        print(f"Done: {output} ({output.stat().st_size // 1024} KB)")
        return output
    finally:
        _stop_servers(spawned)
        if VIDEO_TMP.exists():
            shutil.rmtree(VIDEO_TMP, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record E2E product demo MP4")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found — install with: brew install ffmpeg")

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    record_demo(args.query, args.output)


if __name__ == "__main__":
    main()
