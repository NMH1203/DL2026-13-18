"""Stable command-line entry point for the Project 18 pipeline.

Keep this file intentionally small. Contributor-specific implementation lives
under ``src/<contributor>/`` so team members can work without repeatedly
editing the same orchestration file.
"""

from src.luong.pipeline import main


if __name__ == "__main__":
    main()
