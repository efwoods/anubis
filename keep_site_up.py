# visit_neural_nexus.py

from playwright.sync_api import sync_playwright
from datetime import datetime

URL = "https://neural-nexus-ui.streamlit.app/?assistant_id=cd8ddcc4-6051-4adb-8876-231e0f3a7105"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-dev-shm-usage",
            "--no-sandbox"
        ]
    )

    page = browser.new_page()

    page.goto(
        URL,
        wait_until="networkidle",
        timeout=120000
    )

    print(
        f"{datetime.utcnow().isoformat()} UTC "
        f"visited successfully"
    )

    browser.close()
