import { afterEach, describe, expect, it, vi } from "vitest";
import { MAX_UPLOAD_BYTES } from "../../static/js/upload_helpers.js";

const renderDom = (apiBase = "http://example.com") => {
  document.body.innerHTML = `
    <p id="warmupMessage"></p>
    <div id="uploadSection"></div>
    <div id="resultsSection" style="display:none"></div>
    <form id="uploadForm">
      <input type="file" id="fileInput" />
      <label for="fileInput">Choose File</label>
      <input type="submit" id="submitBtn" />
      <span id="loadingSpinner"></span>
      <p id="statusMessage"></p>
      <span id="fileName">No file chosen</span>
      <div id="previewSection" class="is-hidden">
        <img id="previewImage" />
      </div>
    </form>
    <p id="bigResultsText" class="is-hidden"></p>
    <p id="smallResultsText" class="is-hidden"></p>
    <div id="resultImages"></div>
    <button id="tryAnotherBtn" type="button"></button>
    <button id="tryAnotherTopBtn" type="button"></button>
    <div id="buildVersion"></div>
  `;
  document.body.dataset.apiBase = apiBase;
};

const setInputFiles = (input, files) => {
  Object.defineProperty(input, "files", {
    value: files,
    writable: false,
    configurable: true,
  });
};

const flushPromises = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

const mockFetchResponse = (ok, payload, status = ok ? 200 : 500) => ({
  ok,
  status,
  json: async () => payload,
});

const setup = async ({ apiBase = "http://example.com", autoWarmup = false, fetchMock } = {}) => {
  renderDom(apiBase);
  window.AGE_PREDICT_DISABLE_AUTO_INIT = true;
  window.scrollTo = vi.fn();
  window.URL.createObjectURL = vi.fn(() => "blob://preview");
  window.URL.revokeObjectURL = vi.fn();
  window.fetch = fetchMock || vi.fn();
  vi.resetModules();
  const module = await import("../../static/js/upload.js");
  module.initUpload({ root: document, autoWarmup });
};

describe("upload DOM wiring", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    delete window.AGE_PREDICT_DISABLE_AUTO_INIT;
    delete window.heic2any;
  });

  it("disables submit and clears file name when no file selected", async () => {
    await setup();
    const fileInput = document.getElementById("fileInput");
    const submitBtn = document.getElementById("submitBtn");
    const fileName = document.getElementById("fileName");

    setInputFiles(fileInput, []);
    fileInput.dispatchEvent(new Event("change"));

    expect(submitBtn.disabled).toBe(true);
    expect(fileName.textContent).toBe("No file chosen");
  });

  it("shows an error when file is too large", async () => {
    await setup();
    const fileInput = document.getElementById("fileInput");
    const submitBtn = document.getElementById("submitBtn");
    const statusMessage = document.getElementById("statusMessage");
    const bigBuffer = new Uint8Array(MAX_UPLOAD_BYTES + 1);
    const bigFile = new File([bigBuffer], "big.jpg", { type: "image/jpeg" });

    setInputFiles(fileInput, [bigFile]);
    fileInput.dispatchEvent(new Event("change"));

    expect(submitBtn.disabled).toBe(true);
    expect(statusMessage.textContent).toContain("File too large");
  });

  it("allows a file exactly at the max upload size", async () => {
    await setup();
    const fileInput = document.getElementById("fileInput");
    const submitBtn = document.getElementById("submitBtn");
    const statusMessage = document.getElementById("statusMessage");
    const exactBuffer = new Uint8Array(MAX_UPLOAD_BYTES);
    const file = new File([exactBuffer], "exact.jpg", { type: "image/jpeg" });

    setInputFiles(fileInput, [file]);
    fileInput.dispatchEvent(new Event("change"));
    await flushPromises();

    expect(submitBtn.disabled).toBe(false);
    expect(statusMessage.textContent).toBe("");
  });

  it("Try Another resets UI and scrolls to top", async () => {
    await setup();
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const resultsSection = document.getElementById("resultsSection");
    const uploadSection = document.getElementById("uploadSection");
    const resultImages = document.getElementById("resultImages");
    const tryAnotherBtn = document.getElementById("tryAnotherBtn");

    const selected = new File(["x"], "photo.jpg", { type: "image/jpeg" });
    setInputFiles(fileInput, [selected]);
    fileName.textContent = "photo.jpg";
    previewImage.src = "blob://preview";
    previewSection.classList.remove("is-hidden");
    resultsSection.style.display = "block";
    uploadSection.style.display = "none";
    resultImages.innerHTML = "<img src='x' />";

    tryAnotherBtn.dispatchEvent(new Event("click"));

    expect(fileInput.value).toBe("");
    expect(fileName.textContent).toBe("No file chosen");
    expect(previewSection.classList.contains("is-hidden")).toBe(true);
    expect(resultsSection.style.display).toBe("none");
    expect(uploadSection.style.display).toBe("");
    expect(window.scrollTo).toHaveBeenCalled();
  });

  it("Try Another (top button) resets UI and scrolls to top", async () => {
    await setup();
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const resultsSection = document.getElementById("resultsSection");
    const uploadSection = document.getElementById("uploadSection");
    const tryAnotherTopBtn = document.getElementById("tryAnotherTopBtn");

    const selected = new File(["x"], "photo.jpg", { type: "image/jpeg" });
    setInputFiles(fileInput, [selected]);
    fileName.textContent = "photo.jpg";
    previewImage.src = "blob://preview";
    previewSection.classList.remove("is-hidden");
    resultsSection.style.display = "block";
    uploadSection.style.display = "none";

    tryAnotherTopBtn.dispatchEvent(new Event("click"));

    expect(fileName.textContent).toBe("No file chosen");
    expect(previewSection.classList.contains("is-hidden")).toBe(true);
    expect(resultsSection.style.display).toBe("none");
    expect(uploadSection.style.display).toBe("");
    expect(window.scrollTo).toHaveBeenCalled();
  });

  it("submit without file shows a message", async () => {
    await setup();
    const form = document.getElementById("uploadForm");
    const statusMessage = document.getElementById("statusMessage");

    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(statusMessage.textContent).toContain("Please choose an image");
  });

  it("submit with same-origin apiBase uses relative /api path", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(mockFetchResponse(true, { apiBase: "", buildVersion: "test" }))
      .mockResolvedValueOnce(mockFetchResponse(true, { key: "obj", url: "http://upload.local" }))
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce(mockFetchResponse(true, { fig_urls: [] }));
    await setup({ apiBase: "", autoWarmup: false, fetchMock });
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith("./config.json", { cache: "no-store" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/presign",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("submit with oversized file is blocked", async () => {
    await setup({ autoWarmup: false });
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");
    const bigBuffer = new Uint8Array(MAX_UPLOAD_BYTES + 1);
    const bigFile = new File([bigBuffer], "big.jpg", { type: "image/jpeg" });

    setInputFiles(fileInput, [bigFile]);
    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(statusMessage.textContent).toContain("File too large");
  });

  it("warmup with same-origin apiBase calls relative health endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(mockFetchResponse(true, {}));
    await setup({ apiBase: "", autoWarmup: true, fetchMock });
    await flushPromises();

    const submitBtn = document.getElementById("submitBtn");
    expect(submitBtn.disabled).toBe(false);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/health?deep=true",
      expect.objectContaining({ cache: "no-store" })
    );
  });

  it("warmup disables and re-enables submit while calling health endpoint", async () => {
    let resolveHealth;
    const healthPromise = new Promise((resolve) => {
      resolveHealth = resolve;
    });
    const fetchMock = vi.fn().mockImplementation(() => healthPromise);
    await setup({ autoWarmup: true, fetchMock });

    const submitBtn = document.getElementById("submitBtn");
    expect(submitBtn.disabled).toBe(true);

    resolveHealth(mockFetchResponse(true, {}));
    await flushPromises();

    expect(submitBtn.disabled).toBe(false);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://example.com/api/health?deep=true",
      expect.objectContaining({ cache: "no-store" })
    );
  });

  it("retry flow: first predict fails retryable then succeeds and renders results", async () => {
    vi.useFakeTimers();
    try {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(mockFetchResponse(true, { key: "obj", url: "http://upload.local" }))
        .mockResolvedValueOnce({ ok: true })
        .mockResolvedValueOnce(mockFetchResponse(false, { message: "503" }, 503))
        .mockResolvedValueOnce(
          mockFetchResponse(true, {
            big_fig_url: "http://img/big.png",
            fig_urls: ["http://img/s1.png"],
          })
        );
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

      await setup({ autoWarmup: false, fetchMock });
      const form = document.getElementById("uploadForm");
      const fileInput = document.getElementById("fileInput");
      const warmupMessage = document.getElementById("warmupMessage");
      const resultsSection = document.getElementById("resultsSection");
      const resultImages = document.getElementById("resultImages");

      setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
      form.dispatchEvent(new Event("submit"));
      await flushPromises();
      await vi.advanceTimersByTimeAsync(5000);
      await flushPromises();

      expect(warmupMessage.textContent).toBe("");
      expect(resultsSection.style.display).toBe("block");
      expect(resultImages.querySelectorAll("img").length).toBe(2);
      expect(fetchMock).toHaveBeenCalledTimes(4);
      expect(fetchMock.mock.calls[0][0]).toBe("http://example.com/api/presign");
      expect(fetchMock.mock.calls[1][0]).toBe("http://upload.local");
      expect(fetchMock.mock.calls[2][0]).toBe("http://example.com/api/predict");
      expect(fetchMock.mock.calls[3][0]).toBe("http://example.com/api/predict");
      expect(errorSpy).not.toHaveBeenCalled();
      errorSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });

  it("HEIC converter load failure shows specific message", async () => {
    const fetchMock = vi.fn();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    window.heic2any = undefined;
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    setInputFiles(fileInput, [new File(["x"], "face.heic", { type: "image/heic" })]);
    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(statusMessage.textContent).toContain("HEIC conversion library failed to load");
    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("backend payload message is surfaced on submit error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(mockFetchResponse(false, { message: "Face not found" }, 400));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await vi.waitFor(() => {
      expect(statusMessage.textContent).toBe("Face not found");
    });

    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("generic submit errors show fallback message", async () => {
    const fetchMock = vi.fn().mockRejectedValueOnce(new Error("network down"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await vi.waitFor(() => {
      expect(statusMessage.textContent).toContain("Something went wrong. Please try again.");
    });

    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("non-JSON backend errors fall back to generic message", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    });
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await vi.waitFor(() => {
      expect(statusMessage.textContent).toContain("Something went wrong. Please try again.");
    });

    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("renders small result images when no big figure is returned", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(mockFetchResponse(true, { key: "obj", url: "http://upload.local" }))
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce(mockFetchResponse(true, { fig_urls: ["http://img/s1.png", "http://img/s2.png"] }));

    await setup({ autoWarmup: false, fetchMock });
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const bigResultsText = document.getElementById("bigResultsText");
    const smallResultsText = document.getElementById("smallResultsText");
    const resultImages = document.getElementById("resultImages");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await vi.waitFor(() => {
      expect(resultImages.querySelectorAll("img").length).toBe(2);
    });

    expect(bigResultsText.classList.contains("is-hidden")).toBe(true);
    expect(smallResultsText.classList.contains("is-hidden")).toBe(false);
  });

  it("toggles submit button and spinner visibility during submit lifecycle", async () => {
    let resolvePresign;
    const presignPromise = new Promise((resolve) => {
      resolvePresign = resolve;
    });
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => presignPromise)
      .mockResolvedValueOnce({ ok: false });
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const fileButton = document.querySelector("label[for='fileInput']");
    const submitBtn = document.getElementById("submitBtn");
    const spinner = document.getElementById("loadingSpinner");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(submitBtn.style.display).toBe("none");
    expect(spinner.style.display).toBe("inline-block");
    expect(fileInput.disabled).toBe(true);
    expect(fileButton.style.display).toBe("none");

    resolvePresign(mockFetchResponse(true, { key: "obj", url: "http://upload.local" }));
    await flushPromises();
    await flushPromises();

    expect(submitBtn.style.display).toBe("inline-block");
    expect(spinner.style.display).toBe("none");
    expect(fileInput.disabled).toBe(false);
    expect(fileButton.style.display).toBe("inline-block");
    errorSpy.mockRestore();
  });

  it("ignores duplicate submit events while request is in flight", async () => {
    let resolvePresign;
    const presignPromise = new Promise((resolve) => {
      resolvePresign = resolve;
    });
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => presignPromise)
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce(mockFetchResponse(true, { fig_urls: [] }));
    await setup({ autoWarmup: false, fetchMock });

    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");

    setInputFiles(fileInput, [new File(["x"], "photo.jpg", { type: "image/jpeg" })]);
    form.dispatchEvent(new Event("submit"));
    form.dispatchEvent(new Event("submit"));
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("http://example.com/api/presign");

    resolvePresign(mockFetchResponse(true, { key: "obj", url: "http://upload.local" }));
    await flushPromises();
  });
});
