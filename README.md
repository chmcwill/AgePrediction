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

## Frontend Config
The frontend reads `static/config.json`.
- In deployed mode, keep `apiBase` as `""` to use same-origin `/api/*` via CloudFront.
- `scripts/redeploy_all.ps1` refreshes `buildVersion` during deploy.

Note: `static/index.html` references `js/upload.js?v=1` for cache busting. When you
change frontend JS, bump the query string value if you want browsers to fetch the new file.

## HEIC/HEIF Uploads
HEIC/HEIF files are converted to JPG in the browser via a vendored `heic2any` script in `static/js/heic2any.min.js` (no CDN dependency). The backend does not require `libheif` in Lambda, which keeps the image build simpler.

## Tests
Run all tests:
```powershell
.\scripts\run_all_tests.ps1
.\scripts\run_all_tests.ps1 -PythonExe C:\Users\camer\anaconda3\envs\flaskapplambda\python.exe
```

Unit tests:
```bash
python -m pytest -q
```

Integration tests (real models + real images):
```bash
python -m pytest -q -m integration
```

Note: Integration tests depend on `best_models/` weights and real face detection. They can be fragile if detector thresholds or model weights change, so expect occasional updates when the pipeline changes.

Frontend tests:
```bash
npm test
```

Frontend tests in watch mode:
```bash
npm run test:watch
```

E2E tests (Playwright):
```bash
npm run test:e2e:install
npm run test:e2e
```

Run only mocked upload e2e:
```bash
npm run test:e2e:mocked
```

Run only full-stack e2e:
```bash
npm run test:e2e:fullstack
```

## GitHub CI/CD (Starter Setup)
This repo now uses two GitHub Actions workflows:

- `.github/workflows/ci.yml`
  - Triggers on push (`dev`, `main`) and all pull requests.
  - Runs backend tests (`pytest -m "not integration"`), frontend unit tests (`vitest`), and mocked Playwright e2e.
- `.github/workflows/deploy-dev.yml`
  - Manual trigger (`workflow_dispatch`) for controlled deploys.
  - Builds/pushes `Dockerfile.lambda` to ECR, deploys `template.yaml`, syncs `static/`, invalidates CloudFront.

### 1) Configure GitHub Variables
Go to: `GitHub repo -> Settings -> Secrets and variables -> Actions -> Variables`

Required:
- `AWS_REGION` (example: `us-east-2`)
- `AWS_ACCOUNT_ID` (your 12-digit AWS account id)
- `ECR_REPOSITORY` (example: `agepred-predict-age`)

Optional (defaults are baked into workflow):
- `STACK_NAME` (default: `agepred-serverless`)
- `BUCKET_PREFIX` (default: `agepred`)
- `FUNCTION_MEMORY_SIZE` (default: `3008`)
- `FUNCTION_TIMEOUT` (default: `60`)
- `PRESIGN_EXPIRE_SECONDS` (default: `600`)
- `RESULT_URL_EXPIRE_SECONDS` (default: `3600`)
- `API_CUSTOM_DOMAIN_NAME` (default: `api.facepredictionservice.com`)
- `API_CUSTOM_DOMAIN_CERT_ARN` (blank unless custom domain is enabled)

### 2) Configure GitHub Secrets for AWS Auth
Preferred (OIDC):
- Add secret `AWS_ROLE_TO_ASSUME` with an IAM role ARN trusted by GitHub OIDC.

Fallback (access keys):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### 3) How to use it
- CI: open a PR or push to `dev`/`main`; checks run automatically in Actions.
- CD: open `Actions -> Deploy Dev (Serverless) -> Run workflow` when you want to deploy.

### 4) Recommended branch policy
- Protect `main`.
- Require `CI` workflow to pass before merge.
- Keep deploys manual until you are comfortable, then optionally switch deploy trigger to push on `main` or `dev`.
