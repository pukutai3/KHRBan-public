"""Japanese GUI launcher for KHR policy playback."""

from __future__ import annotations

from khrban.viser_ja import install_japanese_viser_labels


def main() -> None:
    install_japanese_viser_labels()

    # Importing the task package registers KHR task IDs with MjLab.
    import khrban.tasks  # noqa: F401
    from mjlab.scripts.play import main as mjlab_play_main

    mjlab_play_main()


if __name__ == "__main__":
    main()
