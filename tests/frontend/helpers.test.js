import { describe, expect, it } from "vitest";
import {
  MAX_UPLOAD_BYTES,
  convertHeicToJpeg,
  isHeicFile,
  shouldRetryPredict,
} from "../../static/js/upload_helpers.js";

describe("upload_helpers", () => {
  it("shouldRetryPredict returns true for retryable errors", () => {
    const err = new Error("502 bad gateway");
    expect(shouldRetryPredict(err)).toBe(true);
    expect(shouldRetryPredict(new Error("timeout"))).toBe(true);
    expect(shouldRetryPredict(new Error("Internal Server Error"))).toBe(true);
    expect(shouldRetryPredict(new Error("503 unavailable"))).toBe(true);
  });

  it("shouldRetryPredict returns false for non-retryable errors", () => {
    expect(shouldRetryPredict(new Error("bad request"))).toBe(false);
    expect(shouldRetryPredict(new Error("400 invalid image"))).toBe(false);
  });

  it("isHeicFile detects HEIC/HEIF files by name or type", () => {
    const heicByType = new File(["x"], "photo.jpg", { type: "image/heic" });
    const heifByName = new File(["x"], "photo.HEIF", { type: "image/jpeg" });
    const jpg = new File(["x"], "photo.jpg", { type: "image/jpeg" });
    expect(isHeicFile(heicByType)).toBe(true);
    expect(isHeicFile(heifByName)).toBe(true);
    expect(isHeicFile(jpg)).toBe(false);
  });

  it("convertHeicToJpeg uses injected converter and returns jpeg file", async () => {
    const fakeHeic2Any = async () => new Blob(["jpeg"], { type: "image/jpeg" });
    const file = new File(["heic"], "image.heic", { type: "image/heic" });
    const converted = await convertHeicToJpeg(file, fakeHeic2Any);
    expect(converted.name).toBe("image.jpg");
    expect(converted.type).toBe("image/jpeg");
  });

  it("convertHeicToJpeg throws when converter is missing", async () => {
    const file = new File(["heic"], "image.heic", { type: "image/heic" });
    await expect(convertHeicToJpeg(file, null)).rejects.toThrow("heic2any_not_loaded");
  });

  it("MAX_UPLOAD_BYTES matches 10MB cap", () => {
    expect(MAX_UPLOAD_BYTES).toBe(10 * 1024 * 1024);
    expect(MAX_UPLOAD_BYTES + 1).toBe(10 * 1024 * 1024 + 1);
  });
});
