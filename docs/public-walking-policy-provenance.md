# 公開歩行ポリシーの根拠

## 欠落の再現と原因

- Issue ID: `PUBLIC-WALK-POLICY-001`
- 公開版の変更前commit: `bd693313606d814295782b83670e6b3797ea727e`
- 現象: 公開版に学習済み `.pt` / `.onnx` がなく、新規取得した利用者は歩行ポリシーを再生できない。
- 再現: `tests/test_public_policy.py::test_evaluated_walking_policy_is_distributed` が、`policies/velocity/model_179910.pt` 不在で失敗。
- 原因: ソースだけを公開版へ移し、学習成果を保存する `logs/` は `.gitignore` で除外したまま、合格チェックポイントの配布先を作らなかった。

## 配布候補の選定

- 元の学習run: `2026-09-30_06-14-34_khr-knee-ankle-055-20260924-0427624-20260923T173842488356Z-round-0135`
- 元の学習commit: `fc11b71791d7abe2219c0756582ad95d57855017`
- チェックポイント: `model_179910.pt`（4,899,737 bytes）
- 元ファイルのSHA-256: `719f01fa07eb53dab2d0fb03af5e78869e10c6845dc1f7f9fb993baf2b133185`
- 選定根拠: 自動評価履歴の契約version 4で `achieved: true`。64環境×500ステップの評価で、丸め後の直線速度RMSEは0.10 m/s、ヨー速度RMSEは0.24 rad/s、方向別RMSEは0.12 m/s、純旋回ヨーRMSEは0.26 rad/s、足上げ成功率は0.50だった。

これはシミュレーション上の評価であり、実機での安全性・性能を証明しない。公開版のコード・モデルでの再評価および配布物の動作確認結果は、検証後に追記する。
