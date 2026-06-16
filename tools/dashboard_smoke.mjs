import fs from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

const dashboardUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8765";
const screenshotPath = process.env.DASHBOARD_SCREENSHOT_PATH
  || path.resolve("data/runtime/dashboard-chrome-smoke.png");
const chromeExecutable = process.env.CHROME_EXECUTABLE_PATH
  || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await chromium.launch({
  headless: true,
  executablePath: chromeExecutable,
});

try {
  const page = await browser.newPage({
    viewport: { width: 1480, height: 1600 },
    locale: "ko-KR",
  });

  await page.goto(dashboardUrl, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForSelector("[data-testid='ops-overview']");
  await page.waitForSelector("[data-testid='trade-readiness-card']");
  await page.waitForSelector("[data-testid='runtime-state-value']");
  await page.waitForSelector("[data-testid='risk-cap-value']");
  await page.waitForSelector("[data-testid='performance-summary-card']");
  await page.waitForSelector("[data-testid='recent-trade-ledger-card']");

  const snapshot = {
    title: await page.locator("[data-testid='page-title']").innerText(),
    readiness: await page.locator("#readinessValue").innerText(),
    runtime: await page.locator("[data-testid='runtime-state-value']").innerText(),
    risk: await page.locator("[data-testid='risk-cap-value']").innerText(),
    performance: await page.locator("[data-testid='performance-summary-card'] h3").innerText(),
  };

  await fs.mkdir(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });

  console.log(JSON.stringify({
    status: "ok",
    dashboardUrl,
    screenshotPath,
    snapshot,
  }, null, 2));
} finally {
  await browser.close();
}
