#!/usr/bin/env python
"""Capture UI screenshots of the running app into docs/screenshots/.

Requires the API (port 8000) and the UI (default http://127.0.0.1:5173) to be
running. Uses Playwright driving an existing Chromium:

    uv run --with playwright python scripts/screenshots.py [--base URL] [--chromium PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
MODE = {"text": "Text", "image": "Image", "hybrid": "Hybrid"}


def sample_image(collection: str, label: str | None = None) -> Path:
    con = sqlite3.connect(ROOT / "data" / "catalog.sqlite")
    q = "SELECT path FROM images WHERE collection=? AND deleted=0"
    args: list = [collection]
    if label:
        q += " AND label=?"
        args.append(label)
    q += " AND split IN ('test','val') ORDER BY id LIMIT 1 OFFSET 7"
    (path,) = con.execute(q, args).fetchone()
    return ROOT / "data" / "images" / path


def wait_results(page: Page) -> None:
    page.wait_for_selector(".card-hit", timeout=60_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(600)


def fresh(page: Page, url: str) -> None:
    """Navigate and reload: views keep their state across hash changes."""
    page.goto(url)
    page.reload()
    page.wait_for_load_state("networkidle")


def text_search(page: Page, base: str, query: str) -> None:
    fresh(page, f"{base}/#/search")
    page.get_by_role("radio", name=MODE["text"]).first.click()
    box = query_box(page)
    box.fill(query)
    box.press("Enter")
    wait_results(page)


def query_box(page: Page):
    return page.get_by_label("Describe the image you are looking for").first


def shot(page: Page, name: str, full: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("saved", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5173")
    ap.add_argument("--chromium", default="/usr/bin/chromium")
    args = ap.parse_args()
    base = args.base
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(viewport={"width": 1440, "height": 950}, color_scheme="light")
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        text_search(page, base, "a red backpack placed on a wooden table near a window")
        shot(page, "01_text_search")
        text_search(page, base, "two dogs playing in the snow")
        shot(page, "02_text_search_dogs")

        page.locator(".card-hit").first.click()
        page.wait_for_timeout(1200)
        shot(page, "03_detail_modal")
        page.keyboard.press("Escape")

        fresh(page, f"{base}/#/search")
        page.get_by_role("radio", name=MODE["image"]).first.click()
        page.locator("input[type=file]").first.set_input_files(str(sample_image("imagewoof", "Samoyed")))
        page.get_by_role("button", name="Search").last.click()
        wait_results(page)
        shot(page, "04_image_search_dinov2")

        fresh(page, f"{base}/#/search")
        page.get_by_role("radio", name=MODE["hybrid"]).first.click()
        page.locator("input[type=file]").first.set_input_files(str(sample_image("imagewoof", "golden retriever")))
        box = query_box(page)
        box.fill("a dog swimming in a lake")
        box.press("Enter")
        wait_results(page)
        shot(page, "05_hybrid_search")

        text_search(page, base, "a man riding a bicycle on a city street")
        for name in ("Options", "Filters"):
            btn = page.get_by_role("button", name=name)
            if btn.count():
                btn.first.click()
        page.wait_for_timeout(500)
        shot(page, "06_options_and_filters")

        fresh(page, f"{base}/#/compare")
        box = query_box(page)
        box.fill("a little girl in a pink dress climbing stairs")
        box.press("Enter")
        page.wait_for_timeout(8000)
        shot(page, "07_compare")

        fresh(page, f"{base}/#/experiments")
        page.wait_for_timeout(2500)
        shot(page, "08_experiments", full=True)

        fresh(page, f"{base}/#/system")
        page.wait_for_timeout(2500)
        shot(page, "09_system", full=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="dark",
                                  device_scale_factor=2)
        text_search(mobile, base, "a golden retriever lying on grass")
        shot(mobile, "10_mobile_dark")
        browser.close()
        if errors:
            print("console errors:", *errors[:10], sep="\n  ")


if __name__ == "__main__":
    main()
