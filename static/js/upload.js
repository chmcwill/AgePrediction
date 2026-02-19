import {
  API_TIMEOUT_MS,
  WARMUP_TIMEOUT_MS,
  MAX_UPLOAD_MB,
  MAX_UPLOAD_BYTES,
  shouldRetryPredict,
  isHeicFile,
  convertHeicToJpeg,
} from "./upload_helpers.js";

export const initUpload = (options = {}) => {
  const doc = options.root || document;
  // Browser flow: presign -> direct upload to S3 -> predict -> show results.
  const form = doc.getElementById("uploadForm");
  if (!form) {
    return;
  }

  const fileInput = doc.getElementById("fileInput");
  const submitBtn = doc.getElementById("submitBtn");
  const spinner = doc.getElementById("loadingSpinner");
  const statusMessage = doc.getElementById("statusMessage");
  const warmupMessage = doc.getElementById("warmupMessage");
  const buildVersion = doc.getElementById("buildVersion");
  const fileName = doc.getElementById("fileName");
  const previewSection = doc.getElementById("previewSection");
  const previewImage = doc.getElementById("previewImage");
  const uploadSection = doc.getElementById("uploadSection");
  const resultsSection = doc.getElementById("resultsSection");
  const bigResultsText = doc.getElementById("bigResultsText");
  const resultImages = doc.getElementById("resultImages");
  const smallResultsText = doc.getElementById("smallResultsText");
  const tryAnotherBtn = doc.getElementById("tryAnotherBtn");
  const tryAnotherTopBtn = doc.getElementById("tryAnotherTopBtn");
  let apiBase = window.AGE_PREDICT_API_BASE || doc.body.dataset.apiBase || "";

  // Inputs: isLoading (bool), message (string). Output: UI side effects only.
  const setLoading = (isLoading, message) => {
    if (submitBtn) {
      submitBtn.style.display = isLoading ? "none" : "inline-block";
    }
    if (spinner) {
      spinner.style.display = isLoading ? "inline-block" : "none";
    }
    if (statusMessage) {
      statusMessage.textContent = message || "";
    }
  };

  const setSubmitEnabled = (enabled) => {
    if (submitBtn) {
      submitBtn.disabled = !enabled;
    }
  };

  // Input: none. Output: clears result DOM nodes.
  const clearResults = () => {
    if (resultImages) {
      resultImages.innerHTML = "";
    }
    if (bigResultsText) {
      bigResultsText.classList.add("is-hidden");
    }
    if (smallResultsText) {
      smallResultsText.classList.add("is-hidden");
      smallResultsText.style.display = "";
    }
    if (resultsSection) {
      resultsSection.style.display = "none";
    }
    if (uploadSection) {
      uploadSection.style.display = "";
    }
  };

  const clearPreview = () => {
    if (previewImage) {
      previewImage.src = "";
    }
    if (previewSection) {
      previewSection.classList.add("is-hidden");
    }
  };

  const setFileName = (name) => {
    if (!fileName) {
      return;
    }
    fileName.textContent = name || "No file chosen";
  };

  const setWarmupMessage = (message) => {
    if (!warmupMessage) {
      return;
    }
    if (message) {
      warmupMessage.textContent = message;
      warmupMessage.style.display = "block";
    } else {
      warmupMessage.textContent = "";
      warmupMessage.style.display = "none";
    }
  };

  // Inputs: bigUrl (string|null), figUrls (array). Output: renders images.
  const showResults = (bigUrl, figUrls) => {
    if (!resultsSection || !resultImages) {
      return;
    }
    clearPreview();
    if (uploadSection) {
      uploadSection.style.display = "none";
    }
    resultImages.innerHTML = "";
    const smallUrls = figUrls || [];
    const showBig = Boolean(bigUrl);
    if (bigResultsText) {
      bigResultsText.classList.toggle("is-hidden", !showBig);
      bigResultsText.style.display = "";
    }
    if (showBig) {
      if (bigResultsText) {
        resultImages.appendChild(bigResultsText);
      }
      const bigImg = document.createElement("img");
      bigImg.src = bigUrl;
      bigImg.alt = "Prediction overview";
      bigImg.className = "result-image result-image--big";
      resultImages.appendChild(bigImg);
    }
    if (smallResultsText) {
      const showSmallText = smallUrls.length > 0;
      smallResultsText.classList.toggle("is-hidden", !showSmallText);
      smallResultsText.style.display = "";
      if (showSmallText) {
        resultImages.appendChild(smallResultsText);
      }
    }
    smallUrls.forEach((url) => {
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Prediction detail";
      img.className = "result-image";
      resultImages.appendChild(img);
    });
    resultsSection.style.display = "block";
  };

  // Inputs: path (string), body (object). Output: parsed JSON response.
  const postJson = async (path, body, timeoutMs = API_TIMEOUT_MS) => {
    if (!apiBase) {
      throw new Error("api_base_missing");
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${apiBase}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body || {}),
        signal: controller.signal,
      });
      if (!response.ok) {
        let payload;
        try {
          payload = await response.json();
        } catch (_err) {
          payload = null;
        }
        const message =
          (payload && payload.message) || (payload && payload.error) || "Request failed";
        const error = new Error(message);
        error.status = response.status;
        error.payload = payload;
        throw error;
      }
      return response.json();
    } finally {
      window.clearTimeout(timeoutId);
    }
  };

  const loadApiBaseFromConfig = async () => {
    if (apiBase && apiBase !== "__API_BASE_URL__") {
      return apiBase;
    }
    try {
      const response = await fetch("./config.json", { cache: "no-store" });
      if (!response.ok) {
        return apiBase;
      }
      const payload = await response.json();
      if (payload && payload.apiBase) {
        apiBase = payload.apiBase;
      }
      if (payload && payload.buildVersion && buildVersion) {
        buildVersion.textContent = `Build: ${payload.buildVersion}`;
      }
    } catch (err) {
      // Best-effort; keep existing apiBase if fetch fails.
      // eslint-disable-next-line no-console
      console.error("Failed to load config.json:", err);
    }
    return apiBase;
  };

  const warmupApi = async () => {
    await loadApiBaseFromConfig();
    if (submitBtn) {
      submitBtn.disabled = true;
    }
    if (!apiBase || apiBase === "__API_BASE_URL__") {
      setWarmupMessage("App is not configured with API URL yet. Check /static/config.json.");
      if (submitBtn) {
        submitBtn.disabled = false;
      }
      return;
    }
    setWarmupMessage("Warming up the serverless container and model (cold start)...");
    const warmupFailedMessage =
      "Warmup failed. Serverless cold start may make the first request slower.";
    try {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), WARMUP_TIMEOUT_MS);
      const response = await fetch(`${apiBase}/api/health?deep=true`, {
        cache: "no-store",
        signal: controller.signal,
      });
      window.clearTimeout(timeoutId);
      if (response.ok) {
        setWarmupMessage("");
      } else {
        setWarmupMessage(warmupFailedMessage);
      }
    } catch (err) {
      setWarmupMessage(warmupFailedMessage);
    }
    if (submitBtn) {
      submitBtn.disabled = false;
    }
  };

  if (options.autoWarmup !== false) {
    warmupApi();
  }

  if (fileInput) {
    fileInput.addEventListener("change", async () => {
      const selected = fileInput.files ? fileInput.files[0] : null;
      setFileName(selected ? selected.name : "");
      if (!selected) {
        setLoading(false, "");
        setSubmitEnabled(false);
      } else if (selected.size > MAX_UPLOAD_BYTES) {
        setLoading(false, `File too large. Max size is ${MAX_UPLOAD_MB} MB.`);
        setSubmitEnabled(false);
      } else {
        setLoading(false, "");
        setSubmitEnabled(true);
      }
      if (!selected || !previewSection || !previewImage) {
        clearPreview();
        return;
      }
      try {
        let previewFile = selected;
        if (isHeicFile(previewFile)) {
          previewFile = await convertHeicToJpeg(previewFile);
        }
        const objectUrl = URL.createObjectURL(previewFile);
        previewImage.src = objectUrl;
        previewSection.classList.remove("is-hidden");
        previewImage.onload = () => URL.revokeObjectURL(objectUrl);
      } catch (_err) {
        clearPreview();
      }
    });
  }

  if (tryAnotherBtn) {
    tryAnotherBtn.addEventListener("click", () => {
      if (fileInput) {
        fileInput.value = "";
      }
      setFileName("");
      clearPreview();
      clearResults();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  if (tryAnotherTopBtn) {
    tryAnotherTopBtn.addEventListener("click", () => {
      if (fileInput) {
        fileInput.value = "";
      }
      setFileName("");
      clearPreview();
      clearResults();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearResults();
    await loadApiBaseFromConfig();
    if (!apiBase || apiBase === "__API_BASE_URL__") {
      setLoading(
        false,
        "App is not configured with API URL yet. Ensure static/config.json is reachable at /static/config.json."
      );
      return;
    }

    let file = fileInput && fileInput.files ? fileInput.files[0] : null;
    if (!file) {
      setLoading(false, "Please choose an image before submitting.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setLoading(false, `File too large. Max size is ${MAX_UPLOAD_MB} MB.`);
      return;
    }

    try {
      if (isHeicFile(file)) {
        setLoading(true, "Converting HEIC image...");
        file = await convertHeicToJpeg(file);
      }
      setLoading(true, "Preparing upload...");
      const presign = await postJson("/api/presign", {
        filename: file.name,
        content_type: file.type || "application/octet-stream",
      });

      setLoading(true, "Uploading image...");
      const uploadResponse = await fetch(presign.url, {
        method: "PUT",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
        },
        body: file,
      });
      if (!uploadResponse.ok) {
        throw new Error("Upload failed");
      }

      setLoading(true, "Running prediction...");
      let prediction;
      try {
        prediction = await postJson("/api/predict", { key: presign.key });
      } catch (predictErr) {
        if (!shouldRetryPredict(predictErr)) {
          throw predictErr;
        }
        setWarmupMessage("Warming up the serverless container and model (cold start)...");
        setLoading(true, "Retrying prediction...");
        await new Promise((resolve) => window.setTimeout(resolve, 5000));
        prediction = await postJson("/api/predict", { key: presign.key });
      }
      showResults(prediction.big_fig_url, prediction.fig_urls || []);
      setLoading(false, "");
      setWarmupMessage("");
    } catch (err) {
      const message = (err && err.message) || "";
      if (message === "heic2any_not_loaded") {
        setLoading(
          false,
          "HEIC conversion library failed to load. Please retry or upload a JPG/PNG instead."
        );
      } else if (err && err.payload && err.payload.message) {
        setLoading(false, err.payload.message);
      } else {
        setLoading(false, "Something went wrong. Please try again.");
      }
      // eslint-disable-next-line no-console
      console.error(err);
    }
  });
};

if (typeof document !== "undefined" && !window.AGE_PREDICT_DISABLE_AUTO_INIT) {
  initUpload();
}
