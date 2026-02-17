# AgePrediction
Age Prediction Website (serverless frontend + Lambda backend)

## Quick UI Preview (Static)
```bash
python -m http.server 8000
```
Open:
```
http://localhost:8000/static/index.html
```

## Local Backend + Frontend (Flask)
Use the helper script to set env vars, update `static/config.json`, start the static server, and run the backend:
```powershell
.\scripts\dev_local.ps1
```

Optional local-only mode (skip S3, store uploads/results in `tmp/`):
```powershell
.\scripts\dev_local.ps1 -LocalStorage
```

Notes:
- The script reads `region` + `BucketPrefix` from `deploy.config.json`.
- It derives bucket names using your AWS account id; pass `-AccountId` if needed.
- Use `-UseFlaskCli` to run via `flask run` instead of `python age_prediction/app.py`.
- Use `-SkipUpdateConfig` if you want to keep `static/config.json` unchanged.
- Local storage mode bypasses S3 and uses `/api/upload/...` for the PUT step.

## Configure the Frontend API URL
The frontend reads `static/config.json` for the API base URL.
Run the helper script to update it after each deploy (API Gateway URL changes per stack):
```powershell
.\scripts\update_frontend.ps1 -StackName agepred-serverless
```
Use this when you only changed frontend config or API base.

Note: `static/index.html` references `js/upload.js?v=1` for cache busting. When you
change frontend JS, bump the query string value if you want browsers to fetch the new file.

## HEIC/HEIF Uploads
HEIC/HEIF files are converted to JPG in the browser via a vendored `heic2any` script in `static/js/heic2any.min.js` (no CDN dependency). The backend does not require `libheif` in Lambda, which keeps the image build simpler.

## Tests
Unit tests:
```bash
python -m pytest -q
```

Integration tests (real models + real images):
```bash
python -m pytest -q -m integration
```

Note: Integration tests depend on `best_models/` weights and real face detection. They can be fragile if detector thresholds or model weights change, so expect occasional updates when the pipeline changes.
