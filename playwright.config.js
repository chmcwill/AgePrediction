const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "./tests/frontend/e2e",
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
  },
  webServer: {
    command: "python -m http.server 4173",
    url: "http://127.0.0.1:4173/static/index.html",
    reuseExistingServer: true,
    timeout: 120000,
  },
});
