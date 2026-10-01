from types import SimpleNamespace

from khrban.viser_ja import (
    _ButtonGroupProxy,
    format_time_ago_ja,
    translate_html,
    translate_label,
    translate_options,
)


def test_primary_viewer_labels_are_translated_to_japanese() -> None:
    assert translate_label("Controls") == "操作"
    assert translate_label("Reset Environment") == "環境をリセット"
    assert translate_label("lin_vel_x") == "前後速度 (m/s)"
    assert translate_label("Checkpoints") == "チェックポイント"
    assert translate_label("Camera") == "カメラ"
    assert translate_label("Environment") == "環境"
    assert translate_label("Hide others") == "他の環境を隠す"
    assert translate_label("upright") == "直立姿勢"


def test_button_group_options_keep_machine_values_separate() -> None:
    labels, machine_values = translate_options(("Sync", "Use Latest"))

    assert labels == ("一覧更新", "最新を使用")
    assert machine_values == {"一覧更新": "Sync", "最新を使用": "Use Latest"}


def test_japanese_button_click_delivers_original_machine_value() -> None:
    class FakeHandle:
        def on_click(self, callback):
            self.callback = callback

    handle = FakeHandle()
    proxy = _ButtonGroupProxy(handle, {"最新を使用": "Use Latest"})
    received = []
    proxy.on_click(lambda event: received.append(event.target.value))

    handle.callback(SimpleNamespace(target=SimpleNamespace(value="最新を使用")))

    assert received == ["Use Latest"]


def test_dynamic_status_html_is_translated() -> None:
    translated = translate_html(
        "<strong>Status:</strong> Running [CAPPED]<br/>"
        "<strong>Steps:</strong> 100"
    )

    assert "状態:" in translated
    assert "実行中" in translated
    assert "[速度制限中]" in translated
    assert "ステップ:" in translated


def test_checkpoint_age_is_formatted_in_japanese() -> None:
    assert format_time_ago_ja(45) == "45秒前"
    assert format_time_ago_ja(180) == "3分前"
    assert format_time_ago_ja(7200) == "2時間前"
