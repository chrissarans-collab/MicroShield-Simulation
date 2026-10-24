"""Environment-driven configuration for the MicroShield backend.

All settings have safe local defaults so `uvicorn app.main:app` works with
no environment file at all. See `.env.example` for production usage.
"""

import os
from typing import List

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]


def _parse_origins(raw: str) -> List[str]:
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins


# Browser origins allowed to call the API (comma-separated env var).
CORS_ALLOW_ORIGINS: List[str] = _parse_origins(
    os.getenv("CORS_ALLOW_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
)

# Development-only escape hatch. Never enable in production.
CORS_ALLOW_ALL: bool = os.getenv("CORS_ALLOW_ALL", "").strip() in {"1", "true", "yes"}
