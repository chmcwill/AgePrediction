(() => {
  // Browser flow: presign -> direct upload to S3 -> predict -> show results.
  const form = document.getElementById("uploadForm");
  if (!form) {
    return;
  }

  const fileInput = document.getElementById("fileInput");
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("loadingSpinner");
  const statusMessage = document.getElementById("statusMessage");
  const resultsSection = document.getElementById("resultsSection");
  const resultImages = document.getElementById("resultImages");
  const apiBase = window.AGE_PREDICT_API_BASE || document.body.dataset.apiBase || "";

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

  // Input: none. Output: clears result DOM nodes.
  const clearResults = () => {
    if (resultImages) {
      resultImages.innerHTML = "";
    }
    if (resultsSection) {
      resultsSection.style.display = "none";
    }
  };

  // Inputs: bigUrl (string|null), figUrls (array). Output: renders images.
  const showResults = (bigUrl, figUrls) => {
    if (!resultsSection || !resultImages) {
      return;
    }
    resultImages.innerHTML = "";
    if (bigUrl) {
      const bigImg = document.createElement("img");
      bigImg.src = bigUrl;
      bigImg.alt = "Prediction overview";
      resultImages.appendChild(bigImg);
    }
    (figUrls || []).forEach((url) => {
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Prediction detail";
      resultImages.appendChild(img);
    });
    resultsSection.style.display = "block";
  };

  // Inputs: path (string), body (object). Output: parsed JSON response.
  const postJson = async (path, body) => {
    const response = await fetch(`${apiBase}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body || {}),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Request failed");
    }
    return response.json();
  };

  const isHeicFile = (file) => {
    if (!file) {
      return false;
    }
    const name = (file.name || "").toLowerCase();
    return (
      file.type === "image/heic" ||
      file.type === "image/heif" ||
      name.endsWith(".heic") ||
      name.endsWith(".heif")
    );
  };

  const convertHeicToJpeg = async (file) => {
    if (!window.heic2any) {
      throw new Error("heic2any library not loaded");
    }
    const blob = await window.heic2any({
      blob: file,
      toType: "image/jpeg",
      quality: 0.9,
    });
    const newName = (file.name || "upload").replace(/\.(heic|heif)$/i, ".jpg");
    return new File([blob], newName, { type: "image/jpeg" });
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearResults();

    let file = fileInput && fileInput.files ? fileInput.files[0] : null;
    if (!file) {
      setLoading(false, "Please choose an image before submitting.");
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
      const prediction = await postJson("/api/predict", { key: presign.key });
      showResults(prediction.big_fig_url, prediction.fig_urls || []);
      setLoading(false, "");
    } catch (err) {
      setLoading(false, "Something went wrong. Please try again.");
      // eslint-disable-next-line no-console
      console.error(err);
    }
  });
})();
