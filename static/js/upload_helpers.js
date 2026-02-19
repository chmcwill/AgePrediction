export const API_TIMEOUT_MS = 28000;
export const WARMUP_TIMEOUT_MS = 28000;
export const MAX_UPLOAD_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

export const shouldRetryPredict = (error) => {
  const message = (error && error.message ? error.message : "").toLowerCase();
  return (
    message.includes("endpoint request timed out") ||
    message.includes("timeout") ||
    message.includes("internal server error") ||
    message.includes("502") ||
    message.includes("503") ||
    message.includes("504")
  );
};

export const isHeicFile = (file) => {
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

export const convertHeicToJpeg = async (file, heic2anyLib) => {
  const heic2any =
    heic2anyLib || (typeof window !== "undefined" ? window.heic2any : null);
  if (!heic2any) {
    throw new Error("heic2any_not_loaded");
  }
  const blob = await heic2any({
    blob: file,
    toType: "image/jpeg",
    quality: 0.9,
  });
  const newName = (file.name || "upload").replace(/\.(heic|heif)$/i, ".jpg");
  return new File([blob], newName, { type: "image/jpeg" });
};
