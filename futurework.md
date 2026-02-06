# Future Work

## Async Inference Pipeline (API Gateway + SQS + Worker Lambda)

- Goal: avoid API Gateway's 29-second synchronous timeout and improve reliability for cold starts/heavier images.
- Approach:
  - Add `POST /jobs` endpoint that accepts image metadata and enqueues a job in SQS.
  - Add a worker Lambda (SQS trigger) that runs inference from S3 upload keys.
  - Store job status/results (for example in DynamoDB + S3 result URLs).
  - Add `GET /jobs/{id}` endpoint for client polling.
  - Update frontend to submit job, show processing state, and poll until complete.
- Outcome:
  - No long-running HTTP request.
  - Better UX under cold start.
  - More production-like architecture for portfolio/hiring discussions.

## Hardware speedups
- raise ram in config to get more CPU, but need to make sure that its either better cpu or if its more threads that those threads are utilized.