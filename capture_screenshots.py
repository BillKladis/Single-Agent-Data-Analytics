"""
Playwright screenshot capture for Analyst Copilot.
Launches the Streamlit app, waits for LLM answers, saves real PNGs.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

PORT = 8502
APP_URL = f"http://localhost:{PORT}"
SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

EXAMPLE_QUESTIONS = [
    "Which region is most profitable?",
    "Is discount hurting profit?",
    "What's the sales trend over time?",
]


def wait_for_app(max_wait: int = 60) -> bool:
    import urllib.request
    for _ in range(max_wait):
        try:
            urllib.request.urlopen(APP_URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def wait_for_answer(page, timeout: int = 120_000) -> None:
    """Wait until the spinner disappears and an assistant message appears."""
    try:
        page.wait_for_selector("[data-testid='stSpinner']", timeout=5000)
    except PlaywrightTimeout:
        pass
    page.wait_for_selector("[data-testid='stSpinner']", state="hidden", timeout=timeout)
    page.wait_for_timeout(2000)


def click_example(page, question: str) -> None:
    btn = page.get_by_role("button", name=question[:30])
    if btn.count() == 0:
        btn = page.get_by_text(question[:20])
    btn.first.click()


def main():
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

    print("Starting Streamlit app...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print(f"Waiting for app on {APP_URL}...")
        if not wait_for_app(max_wait=90):
            print("ERROR: App did not start in time.")
            sys.exit(1)
        print("App ready.")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            page = ctx.new_page()
            page.goto(APP_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)

            # Screenshot 1: App loaded / example question view
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "01_app_home.png"),
                full_page=True,
            )
            print("Saved 01_app_home.png")

            # Screenshot 2: "Which region is most profitable?" with answer + chart
            print(f"Asking: {EXAMPLE_QUESTIONS[0]}")
            click_example(page, EXAMPLE_QUESTIONS[0])
            wait_for_answer(page)
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "02_region_profitability.png"),
                full_page=True,
            )
            print("Saved 02_region_profitability.png")

            # Expand the tool trace and screenshot
            page.wait_for_timeout(1000)
            expanders = page.locator("details")
            if expanders.count() > 0:
                expanders.last.click()
                page.wait_for_timeout(1500)
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "03_tools_trace_expanded.png"),
                full_page=True,
            )
            print("Saved 03_tools_trace_expanded.png")

            # Screenshot 4: "Is discount hurting profit?"
            print(f"Asking: {EXAMPLE_QUESTIONS[1]}")
            click_example(page, EXAMPLE_QUESTIONS[1])
            wait_for_answer(page)
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "04_discount_profit.png"),
                full_page=True,
            )
            print("Saved 04_discount_profit.png")

            # Screenshot 5: "What's the sales trend over time?"
            print(f"Asking: {EXAMPLE_QUESTIONS[2]}")
            click_example(page, EXAMPLE_QUESTIONS[2])
            wait_for_answer(page)
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "05_sales_trend.png"),
                full_page=True,
            )
            print("Saved 05_sales_trend.png")

            browser.close()

    finally:
        proc.terminate()
        proc.wait()

    # Verify file sizes
    print("\n--- Screenshot verification ---")
    all_ok = True
    for png in sorted(SCREENSHOTS_DIR.glob("*.png")):
        size_kb = png.stat().st_size / 1024
        status = "OK" if size_kb > 10 else "FAIL (< 10 KB)"
        if size_kb <= 10:
            all_ok = False
        print(f"  {png.name}: {size_kb:.1f} KB  [{status}]")

    if all_ok:
        print("\nAll screenshots verified. Done.")
    else:
        print("\nSome screenshots are too small — check the app.")
        sys.exit(1)


if __name__ == "__main__":
    main()
