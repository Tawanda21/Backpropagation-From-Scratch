from __future__ import annotations

from pathlib import Path


def main() -> int:
    print("Backpropagation from Scratch")
    print("Entry point: src/main.py")
    print("Project root:", Path(__file__).resolve().parents[1])

    try:
        from scratch.train import main as train_main
    except Exception as exc:  # pragma: no cover - fallback for incomplete implementation
        print("The training module is not implemented yet, so this entry point is ready for wiring.")
        print(f"Import fallback: {exc}")
    else:
        print("Dispatching to scratch.train.main()...")
        train_main()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
