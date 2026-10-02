# Server launcher with optional --warmup flag
import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Nyayik AI Backend Server")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Pre-load embedding model and Qdrant client on startup",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes"
    )

    args = parser.parse_args()

    if args.warmup:
        os.environ["PREWARM_EMBEDDINGS"] = "true"
        print("Warmup enabled: embedding model & Qdrant client will pre-load.")

    uvicorn.run(
        "Backend.app.main:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
