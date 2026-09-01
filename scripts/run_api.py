"""启动 API 服务。

    python scripts/run_api.py            # 读 config.yaml 的 host/port
    python scripts/run_api.py --reload   # 改代码自动重载(开发用)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402


def main() -> None:
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=s.host)
    ap.add_argument("--port", type=int, default=s.port)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn

    print(f"→ http://{args.host}:{args.port}/")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "app"), str(ROOT / "pcb"), str(ROOT / "wafer")] if args.reload else None,
    )


if __name__ == "__main__":
    main()
