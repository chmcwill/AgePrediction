#!/usr/bin/env python3
"""Emit deploy config values for GitHub Actions env/output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def load_config(path: Path) -> Dict[str, str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    params = cfg.get("parameters", {})
    return {
        "aws_region": cfg["region"],
        "stack_name": cfg["stack_name"],
        "bucket_prefix": str(params["BucketPrefix"]),
        "function_memory_size": str(params["FunctionMemorySize"]),
        "function_timeout": str(params["FunctionTimeout"]),
        "presign_expire_seconds": str(params["PresignExpireSeconds"]),
        "result_url_expire_seconds": str(params["ResultUrlExpireSeconds"]),
        "api_custom_domain_name": str(params.get("ApiCustomDomainName", "")),
        "api_custom_domain_cert_arn": str(params.get("ApiCustomDomainCertificateArn", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deploy.config.json")
    parser.add_argument(
        "--emit",
        choices=("deploy-outputs", "aws-region-env"),
        required=True,
    )
    args = parser.parse_args()

    values = load_config(Path(args.config))

    if args.emit == "deploy-outputs":
        for key, value in values.items():
            print(f"{key}={value}")
        print(f"AWS_REGION={values['aws_region']}")
        return 0

    print(f"AWS_REGION={values['aws_region']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
