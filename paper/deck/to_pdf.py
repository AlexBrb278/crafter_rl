"""Print deck.html to a PDF (and PNG previews) with headless Chromium.

    pip install playwright && python -m playwright install chromium
    python paper/deck/to_pdf.py [out.pdf] [--png preview_dir]
"""
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(HERE, "barbu_alexandru.pdf")
png_dir = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("file://" + os.path.join(HERE, "deck.html"), wait_until="networkidle")
    page.wait_for_function("document.body.dataset.math === 'ok'", timeout=30000)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(500)
    page.pdf(path=out, width="1280px", height="720px", print_background=True)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        for i, el in enumerate(page.query_selector_all("section.slide")):
            el.screenshot(path=os.path.join(png_dir, f"slide_{i + 1:02d}.png"))
    browser.close()
print("wrote", out)
