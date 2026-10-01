from pathlib import Path

from khrban.training_viewer import (
    active_velocity_run_name,
    latest_checkpoint,
    playback_arguments,
    resolve_training_checkpoint,
    viewer_is_running,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_active_velocity_run_name_ignores_other_processes(tmp_path: Path) -> None:
    (tmp_path / "10").mkdir()
    (tmp_path / "10" / "cmdline").write_bytes(
        b"python\0khrban-train\0--task\0standing\0--run-name\0wrong\0"
    )
    (tmp_path / "20").mkdir()
    (tmp_path / "20" / "cmdline").write_bytes(
        b"python\0khrban-train\0--task\0velocity\0--run-name\0walk-active\0"
    )

    assert active_velocity_run_name(tmp_path) == "walk-active"


def test_latest_checkpoint_uses_numeric_iteration(tmp_path: Path) -> None:
    for name in ("model_900.pt", "model_1000.pt", "model_20.pt"):
        (tmp_path / name).write_bytes(b"complete")

    assert latest_checkpoint(tmp_path) == tmp_path / "model_1000.pt"


def test_playback_uses_sixteen_environments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1000.pt"

    assert playback_arguments(checkpoint) == [
        "Mjlab-KHR-Velocity-Flat",
        "--checkpoint-file",
        str(checkpoint),
        "--num-envs",
        "16",
        "--viewer",
        "viser",
    ]


def test_checkpoint_resolution_prefers_active_velocity_run(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "30").mkdir(parents=True)
    (proc_root / "30" / "cmdline").write_bytes(
        b"python\0khrban-train\0--task\0velocity\0--iterations\0"
        b"15000\0--run-name\0walk-active\0"
    )
    log_root = tmp_path / "logs"
    active = log_root / "2026-01-01_00-00-00_walk-active"
    unrelated = log_root / "2099-01-01_00-00-00_other"
    active.mkdir(parents=True)
    unrelated.mkdir()
    (active / "model_1200.pt").write_bytes(b"active")
    (unrelated / "model_9999.pt").write_bytes(b"unrelated")

    assert resolve_training_checkpoint(log_root, proc_root) == (
        active / "model_1200.pt"
    )


def test_existing_khr_viser_viewer_is_detected(tmp_path: Path) -> None:
    (tmp_path / "40").mkdir()
    (tmp_path / "40" / "cmdline").write_bytes(
        b"python\0khrban-play\0Mjlab-KHR-Velocity-Flat\0"
        b"--num-envs\0" b"1\0--viewer\0viser\0"
    )

    assert viewer_is_running(tmp_path)


def test_windows_control_desk_targets_live_velocity_training() -> None:
    script = (REPO_ROOT / "tools/windows/KHRBan-GUI.ps1").read_text()

    assert "◇  実学習ビューア" in script
    assert "--live-viewer --viewer-num-envs 16" in script
    assert ".venv/bin/khrban-auto-train" in script
    assert "--keep-checkpoints 3" in script
    assert "khrban-train --task velocity" in script
    assert "--num-envs 1536 --iterations-per-round 2000" in script
    assert "実学習ビューアは学習プロセスと一緒に起動します" in script
    assert "logs\\rsl_rl\\khr_velocity" in script
    assert "'--exec', 'pgrep', '-f'" in script
    assert (
        "khrban-auto-train|python[0-9.]* -m "
        "khrban\\.(auto_train|auto_train_getup|train)"
    ) in script
    assert "'--', 'pgrep', '-f', 'khrban[.-](auto.train|train)'" not in script
    assert script.count("$active = Get-ActiveTraining") == 3
    assert "$trainingProbe = Get-ActiveTraining" in script
    assert "$trainingProbe.ExitCode -gt 1" in script
    assert "本学習を検出したため、歩行スモークの起動を防止しました" in script
    assert "▶  速度追従歩行" in script
    assert "↑  起き上がり開始" in script
    assert "New-Button '▶  速度追従歩行' 16 68 300 48" in script
    assert "New-Button '↑  起き上がり開始' 346 68 300 48" in script
    assert "$trainButton.Add_Click({" in script
    assert "$getupButton.Add_Click({" in script
    assert "--check-velocity-gate" in script
    assert "現在のKHR学習または評価が終了してから起き上がり学習を開始してください。" in script
    assert "実際に学習しているKHRを16体表示" in script
    assert "open_model_viewer.html" not in script


def test_windows_control_desk_separates_training_and_motion_playback() -> None:
    script = (REPO_ROOT / "tools/windows/KHRBan-GUI.ps1").read_text()

    assert "$script:TrainingProcess" in script
    assert "$script:MotionViewerProcess" in script
    assert "$motionButton" in script
    assert "モーション確認" in script
    assert ".venv/bin/khrban-motion-viewer" in script
    assert "--device cpu --port 8081" in script
    assert "http://localhost:8080/" in script
    assert "http://localhost:8081/" in script
    assert "taskkill.exe /PID $script:TrainingProcess.Id" in script
    assert "taskkill.exe /PID $script:MotionViewerProcess.Id" in script


def test_windows_control_desk_keeps_the_wslg_session_alive() -> None:
    script = (REPO_ROOT / "tools/windows/KHRBan-GUI.ps1").read_text()

    assert "function Start-WslKeepAlive" in script
    assert "'--exec', 'sleep', 'infinity'" in script
    assert "-WindowStyle Hidden" in script
    shown_handler = script.split("$form.Add_Shown({", maxsplit=1)[1]
    assert shown_handler.index("Start-WslKeepAlive") < shown_handler.index(
        "& $refreshAction"
    )
    assert "Stop-WslKeepAlive" in script
