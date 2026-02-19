const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { test, expect } = require("@playwright/test");

const localStorageDir = path.resolve(__dirname, "../../../tmp/e2e-fullstack");
const repoRoot = path.resolve(__dirname, "../../../");
const backendBase = "http://127.0.0.1:5001";
let backendProcess;
let backendReady = false;
let backendStderr = "";

// Full-stack startup + model warmup can be slower than mocked E2E.
test.setTimeout(90000);

const cleanLocalStorageDir = () => {
  fs.rmSync(localStorageDir, { recursive: true, force: true });
  fs.mkdirSync(localStorageDir, { recursive: true });
};

const waitForBackend = async (timeoutMs = 20000) => {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(`${backendBase}/api/health`);
      if (response.ok) {
        return true;
      }
    } catch (_err) {
      // Retry while backend boots.
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
};

test.beforeAll(async () => {
  cleanLocalStorageDir();
  backendProcess = spawn(
    "python",
    ["-m", "flask", "--app", "age_prediction.app", "run", "--host", "127.0.0.1", "--port", "5001"],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        LOCAL_STORAGE: "1",
        LOCAL_STORAGE_DIR: localStorageDir,
        PYTHONPATH: repoRoot,
      },
      stdio: "pipe",
    }
  );
  if (backendProcess.stderr) {
    backendProcess.stderr.on("data", (chunk) => {
      backendStderr += chunk.toString();
    });
  }
  backendReady = await waitForBackend();
  if (!backendReady) {
    const details = backendStderr.trim() || "No stderr captured from backend process.";
    throw new Error(
      `Full-stack backend failed to start at ${backendBase}. Ensure Flask is installed in the runner Python environment.\n${details}`
    );
  }
});

test.afterAll(async () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill("SIGTERM");
  }
});

test("full stack: frontend calls real backend for presign/upload/predict", async ({ page }) => {
  await page.route("**/config.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        apiBase: backendBase,
        buildVersion: "e2e-fullstack",
      }),
    });
  });

  await page.goto("/static/index.html");
  await expect(page.locator("#uploadForm")).toBeVisible();

  const beforeFiles = fs.readdirSync(localStorageDir);
  await page.locator("#fileInput").setInputFiles({
    name: "not-a-face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("this is not an actual jpg image"),
  });

  await page.locator("#submitBtn").click();

  await page.waitForFunction(() => {
    const spinnerNode = document.querySelector("#loadingSpinner");
    const statusNode = document.querySelector("#statusMessage");
    const resultsNode = document.querySelector("#resultsSection");
    const spinnerVisible = spinnerNode && window.getComputedStyle(spinnerNode).display !== "none";
    const status = (statusNode && statusNode.textContent ? statusNode.textContent : "").trim();
    const resultsVisible = resultsNode && window.getComputedStyle(resultsNode).display !== "none";
    const loadingStatuses = new Set([
      "Converting HEIC image...",
      "Preparing upload...",
      "Uploading image...",
      "Running prediction...",
      "Retrying prediction...",
    ]);
    const hasFinalStatus = status.length > 0 && !loadingStatuses.has(status);
    return !spinnerVisible && (resultsVisible || hasFinalStatus);
  });

  const afterFiles = fs.readdirSync(localStorageDir);
  expect(afterFiles.length).toBeGreaterThan(beforeFiles.length);

  const statusText = await page.locator("#statusMessage").textContent();
  const resultsVisible = await page.locator("#resultsSection").isVisible();

  expect(resultsVisible || (statusText && statusText.trim().length > 0)).toBeTruthy();
});
