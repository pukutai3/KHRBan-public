"""Japanese labels for the KHR MjLab Viser viewer.

The adapter is installed by KHRBan playback and live-training viewers. It leaves
the upstream MjLab and Viser packages untouched and keeps callback values
separate from labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any


_LABELS = {
    "Controls": "操作",
    "Info": "状態",
    "Simulation": "シミュレーション",
    "Play": "再生",
    "Pause": "一時停止",
    "Step": "1ステップ",
    "Reset Environment": "環境をリセット",
    "Speed": "再生速度",
    "Commands": "指令",
    "Twist": "速度指令",
    "Enable": "指令を有効化",
    "Max lin_vel_x": "前後速度上限 (m/s)",
    "lin_vel_x": "前後速度 (m/s)",
    "Max lin_vel_y": "左右速度上限 (m/s)",
    "lin_vel_y": "左右速度 (m/s)",
    "Max ang_vel_z": "旋回速度上限 (rad/s)",
    "ang_vel_z": "旋回速度 (rad/s)",
    "Zero": "ゼロに戻す",
    "Scene": "シーン",
    "Environment": "環境",
    "Hide others": "他の環境を隠す",
    "Camera": "カメラ",
    "Camera Feeds": "カメラ映像",
    "Visualization": "可視化",
    "Rewards": "報酬",
    "Reward": "報酬",
    "Metrics": "指標",
    "Metric": "指標",
    "Groups": "表示グループ",
    "Checkpoints": "チェックポイント",
    "Checkpoint": "チェックポイント",
    "Debug Viz": "デバッグ表示",
    "Enabled": "有効",
    "All envs": "全環境",
    "Track camera": "カメラ追従",
    "FOV (°)": "視野角 (°)",
    "Plots": "グラフ",
    "Select terms": "表示項目を選択",
    "Filter": "絞り込み",
    "Select": "一括選択",
    "track_linear_velocity": "前後・左右速度追従",
    "track_angular_velocity": "旋回速度追従",
    "upright": "直立姿勢",
    "pose": "基準姿勢",
    "body_ang_vel": "胴体角速度",
    "dof_pos_limits": "関節可動範囲",
    "action_rate_l2": "指令変化",
    "air_time": "足の滞空時間",
    "foot_clearance": "足先クリアランス",
    "foot_swing_height": "遊脚高さ",
    "foot_slip": "足滑り",
    "feet_distance": "両足間隔",
    "no_stepping": "静止時足上げ",
    "self_collisions": "自己衝突",
    "mean_action_acc": "平均指令加速度",
}

_OPTION_LABELS = {
    "Slower": "遅く",
    "1x": "1倍",
    "Faster": "速く",
    "Sync": "一覧更新",
    "Use Latest": "最新を使用",
    "All": "すべて",
    "None": "なし",
}

_HTML_REPLACEMENTS = (
    ("Status:", "状態:"),
    ("Running", "実行中"),
    ("Paused", "一時停止中"),
    ("[CAPPED]", "[速度制限中]"),
    ("Steps:", "ステップ:"),
    ("Speed:", "速度:"),
    ("Target RT:", "目標実時間比:"),
    ("Actual RT:", "実時間比:"),
    ("Error:", "エラー:"),
    ("Source:", "取得元:"),
    ("Local", "ローカル"),
    ("Run:", "実行:"),
    ("Open in W&B", "W&Bで開く"),
    ("Showing terms for environment", "表示中の環境"),
)


def translate_label(label: str | None) -> str | None:
    """Translate one visible GUI label while preserving unknown identifiers."""

    if label is None:
        return None
    return _LABELS.get(label, label)


def translate_options(
    options: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return Japanese labels and their original callback-facing values."""

    labels = tuple(_OPTION_LABELS.get(option, option) for option in options)
    return labels, dict(zip(labels, options, strict=True))


def translate_html(content: str) -> str:
    """Translate stable viewer status fragments embedded in HTML/Markdown."""

    for source, target in _HTML_REPLACEMENTS:
        content = content.replace(source, target)
    return content


def format_time_ago_ja(seconds: int) -> str:
    """Format a checkpoint age for the Japanese checkpoint selector."""

    for divisor, unit in ((86400, "日"), (3600, "時間"), (60, "分")):
        if seconds >= divisor:
            return f"{seconds // divisor}{unit}前"
    return f"{seconds}秒前"


class _TargetProxy:
    def __init__(self, target: Any, value: Any) -> None:
        self._target = target
        self.value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _EventProxy:
    def __init__(self, event: Any, value: Any) -> None:
        self._event = event
        self.target = _TargetProxy(event.target, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._event, name)


class _ButtonGroupProxy:
    def __init__(self, handle: Any, machine_values: dict[str, str]) -> None:
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_machine_values", machine_values)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._handle, name, value)

    def on_click(self, callback: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(callback)
        def translated_callback(event: Any) -> Any:
            value = self._machine_values.get(event.target.value, event.target.value)
            return callback(_EventProxy(event, value))

        self._handle.on_click(translated_callback)
        return callback


_INSTALLED = False


def install_japanese_viser_labels() -> None:
    """Install process-local Japanese labels before the Viser viewer starts."""

    global _INSTALLED
    if _INSTALLED:
        return

    import viser
    from mjlab.viewer.viser import viewer as viewer_module

    viewer_module.format_time_ago = format_time_ago_ja

    label_methods = (
        "add_folder",
        "add_button",
        "add_checkbox",
        "add_slider",
        "add_dropdown",
        "add_text",
        "add_number",
    )
    for method_name in label_methods:
        original = getattr(viser.GuiApi, method_name)

        @wraps(original)
        def localized(
            self: Any,
            label: str | None,
            *args: Any,
            _original=original,
            **kwargs: Any,
        ):
            return _original(self, translate_label(label), *args, **kwargs)

        setattr(viser.GuiApi, method_name, localized)

    original_html = viser.GuiApi.add_html

    @wraps(original_html)
    def localized_html(self: Any, content: str, *args: Any, **kwargs: Any):
        return original_html(self, translate_html(content), *args, **kwargs)

    viser.GuiApi.add_html = localized_html

    original_markdown = viser.GuiApi.add_markdown

    @wraps(original_markdown)
    def localized_markdown(self: Any, content: str, *args: Any, **kwargs: Any):
        return original_markdown(self, translate_html(content), *args, **kwargs)

    viser.GuiApi.add_markdown = localized_markdown

    original_button_group = viser.GuiApi.add_button_group

    @wraps(original_button_group)
    def localized_button_group(
        self: Any,
        label: str,
        options: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> _ButtonGroupProxy:
        labels, machine_values = translate_options(options)
        handle = original_button_group(
            self,
            translate_label(label),
            labels,
            *args,
            **kwargs,
        )
        return _ButtonGroupProxy(handle, machine_values)

    viser.GuiApi.add_button_group = localized_button_group

    original_add_tab = viser.GuiTabGroupHandle.add_tab

    @wraps(original_add_tab)
    def localized_tab(self: Any, label: str, *args: Any, **kwargs: Any):
        return original_add_tab(self, translate_label(label), *args, **kwargs)

    viser.GuiTabGroupHandle.add_tab = localized_tab

    original_sync = viewer_module.ViserPlayViewer._sync_ui_state

    @wraps(original_sync)
    def localized_sync(self: Any) -> None:
        original_sync(self)
        self._pause_button.label = "再生" if self._is_paused else "一時停止"

    viewer_module.ViserPlayViewer._sync_ui_state = localized_sync

    def localized_status(self: Any) -> None:
        status = self.get_status()
        actual_rt = status.actual_realtime
        rt_display = f"{actual_rt:.2f}x" if actual_rt > 0 else "—"
        capped = (
            ' <span style="color:#e74c3c;">[速度制限中]</span>'
            if status.capped
            else ""
        )
        error_line = ""
        if status.last_error:
            last_line = status.last_error.strip().splitlines()[-1]
            error_line = (
                f'<br/><span style="color:#e74c3c;">'
                f"<strong>エラー:</strong> {last_line}</span>"
            )
        state = "一時停止中" if status.paused else "実行中"
        self._status_html.content = f"""
          <div style="font-size: 0.85em; line-height: 1.25;
                      padding: 0 1em 0.5em 1em;">
            <strong>状態:</strong> {state}{capped}<br/>
            <strong>ステップ:</strong> {status.step_count}<br/>
            <strong>速度:</strong> {status.speed_label}<br/>
            <strong>目標実時間比:</strong> {status.target_realtime:.2f}x<br/>
            <strong>実時間比:</strong> {rt_display}
            ({status.smoothed_fps:.0f} FPS){error_line}
          </div>
          """

    viewer_module.ViserPlayViewer._update_status_display = localized_status
    _INSTALLED = True
