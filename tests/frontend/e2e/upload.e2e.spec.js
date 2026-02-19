const { test, expect } = require("@playwright/test");

const setupConfigRoute = async (page, apiBase = "http://api.local") => {
  await page.route("**/config.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ apiBase, buildVersion: "e2e" }),
    });
  });
};

const setupSuccessFlowRoutes = async (page, options = {}) => {
  const {
    healthDelayMs = 0,
    presignDelayMs = 0,
    uploadOk = true,
    predictPayload = { big_fig_url: "http://cdn.local/big.png", fig_urls: ["http://cdn.local/s1.png"] },
  } = options;

  await page.route("http://api.local/api/health?deep=true", async (route) => {
    if (healthDelayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, healthDelayMs));
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });

  await page.route("http://api.local/api/presign", async (route) => {
    if (presignDelayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, presignDelayMs));
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ key: "key1", url: "http://upload.local/key1" }),
    });
  });

  await page.route("http://upload.local/key1", async (route) => {
    await route.fulfill({ status: uploadOk ? 200 : 500, body: "" });
  });

  await page.route("http://api.local/api/predict", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(predictPayload),
    });
  });
};

const gotoApp = async (page) => {
  await page.goto("/static/index.html");
  await expect(page.locator("#uploadForm")).toBeVisible();
};

test("cold-start warmup overlap: user can submit while warmup is in progress", async ({ page }) => {
  await setupConfigRoute(page);
  await setupSuccessFlowRoutes(page, { healthDelayMs: 2500 });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });

  await page.locator("#submitBtn").click();
  await expect(page.locator("#loadingSpinner")).toBeVisible();
  await expect(page.locator("#submitBtn")).toBeHidden();

  await expect(page.locator("#resultsSection")).toBeVisible();
  await expect(page.locator("#resultImages img")).toHaveCount(2);
});

test("submit button hides and spinner shows during submit lifecycle", async ({ page }) => {
  await setupConfigRoute(page);
  await setupSuccessFlowRoutes(page, { presignDelayMs: 1200 });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });

  await page.locator("#submitBtn").click();
  await expect(page.locator("#submitBtn")).toBeHidden();
  await expect(page.locator("#loadingSpinner")).toBeVisible();
  await expect(page.locator("#resultsSection")).toBeVisible();
  await expect(page.locator("#loadingSpinner")).toBeHidden();
  await expect(page.locator("#submitBtn")).toBeHidden();
});

test("presign failure shows backend message", async ({ page }) => {
  await setupConfigRoute(page);
  await page.route("http://api.local/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  await page.route("http://api.local/api/presign", async (route) => {
    await route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({ message: "Presign failed" }),
    });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });
  await page.locator("#submitBtn").click();

  await expect(page.locator("#statusMessage")).toContainText("Presign failed");
});

test("upload failure shows generic fallback", async ({ page }) => {
  await setupConfigRoute(page);
  await setupSuccessFlowRoutes(page, { uploadOk: false });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });
  await page.locator("#submitBtn").click();

  await expect(page.locator("#statusMessage")).toContainText("Something went wrong. Please try again.");
});

test("predict failure shows backend message", async ({ page }) => {
  await setupConfigRoute(page);
  await page.route("http://api.local/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  await page.route("http://api.local/api/presign", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ key: "key1", url: "http://upload.local/key1" }),
    });
  });
  await page.route("http://upload.local/key1", async (route) => {
    await route.fulfill({ status: 200, body: "" });
  });
  await page.route("http://api.local/api/predict", async (route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ message: "Predict failed" }),
    });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });
  await page.locator("#submitBtn").click();

  await expect(page.locator("#statusMessage")).toContainText("Predict failed");
});

test("large file is blocked client-side", async ({ page }) => {
  await setupConfigRoute(page);
  let presignCalls = 0;
  await page.route("http://api.local/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  await page.route("http://api.local/api/presign", async (route) => {
    presignCalls += 1;
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "big.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.alloc(10 * 1024 * 1024 + 1),
  });

  await expect(page.locator("#statusMessage")).toContainText("File too large");
  await expect(page.locator("#submitBtn")).toBeDisabled();
  expect(presignCalls).toBe(0);
});

test("HEIC conversion failure shows specific message", async ({ page }) => {
  await setupConfigRoute(page);
  await setupSuccessFlowRoutes(page);
  await gotoApp(page);

  await page.evaluate(() => {
    window.heic2any = undefined;
  });

  await page.locator("#fileInput").setInputFiles({
    name: "face.heic",
    mimeType: "image/heic",
    buffer: Buffer.from("x"),
  });

  await page.locator("#submitBtn").click();
  await expect(page.locator("#statusMessage")).toContainText(
    "HEIC conversion library failed to load"
  );
});

test("user can recover after a failure by resubmitting successfully", async ({ page }) => {
  await setupConfigRoute(page);
  await page.route("http://api.local/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  let presignCallCount = 0;
  await page.route("http://api.local/api/presign", async (route) => {
    presignCallCount += 1;
    if (presignCallCount === 1) {
      await route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({ message: "Presign failed" }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ key: "key1", url: "http://upload.local/key1" }),
    });
  });
  await page.route("http://upload.local/key1", async (route) => {
    await route.fulfill({ status: 200, body: "" });
  });
  await page.route("http://api.local/api/predict", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        big_fig_url: "http://cdn.local/big.png",
        fig_urls: ["http://cdn.local/s1.png"],
      }),
    });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });
  await page.locator("#submitBtn").click();
  await expect(page.locator("#statusMessage")).toContainText("Presign failed");

  await page.locator("#fileInput").setInputFiles({
    name: "face2.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("y"),
  });
  await page.locator("#submitBtn").click();

  await expect(page.locator("#statusMessage")).toHaveText("");
  await expect(page.locator("#resultsSection")).toBeVisible();
  await expect(page.locator("#resultImages img")).toHaveCount(2);
});

test("in-flight submit locks file chooser and ignores duplicate submit", async ({ page }) => {
  await setupConfigRoute(page);
  await page.route("http://api.local/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  let presignCalls = 0;
  await page.route("http://api.local/api/presign", async (route) => {
    presignCalls += 1;
    await new Promise((resolve) => setTimeout(resolve, 1200));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ key: "key1", url: "http://upload.local/key1" }),
    });
  });
  await page.route("http://upload.local/key1", async (route) => {
    await route.fulfill({ status: 500, body: "" });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });

  await page.locator("#submitBtn").click();
  await page.locator("#uploadForm").dispatchEvent("submit");

  await expect(page.locator("#loadingSpinner")).toBeVisible();
  await expect(page.locator("label[for='fileInput']")).toBeHidden();
  await expect(page.locator("#fileInput")).toBeDisabled();
  await expect(page.locator("#statusMessage")).toContainText("Something went wrong. Please try again.");
  await expect(page.locator("label[for='fileInput']")).toBeVisible();
  await expect(page.locator("#fileInput")).toBeEnabled();
  expect(presignCalls).toBe(1);
});

test("same-origin config uses relative /api routes", async ({ page }) => {
  await setupConfigRoute(page, "");
  await page.route("**/api/health?deep=true", async (route) => {
    await route.fulfill({ status: 200, body: "{}" });
  });
  await page.route("**/api/presign", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ key: "key1", url: "http://upload.local/key1" }),
    });
  });
  await page.route("http://upload.local/key1", async (route) => {
    await route.fulfill({ status: 200, body: "" });
  });
  await page.route("**/api/predict", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ big_fig_url: "http://cdn.local/big.png", fig_urls: [] }),
    });
  });
  await gotoApp(page);

  await page.locator("#fileInput").setInputFiles({
    name: "face.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("x"),
  });
  await page.locator("#submitBtn").click();
  await expect(page.locator("#resultsSection")).toBeVisible();
});
