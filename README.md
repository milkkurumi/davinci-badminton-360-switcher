# DaVinci Resolve バドミントン専用360°動画 自動2画角スイッチャー

360°カメラ（DJI Osmo 360）の映像から、TrackNetV3 でシャトルの位置を解析し、
DaVinci Resolve 上で手前コートと奥コートの2画角を自動スイッチするPythonスクリプトです。

---

## ファイル構成

```
/
├── badminton_switcher.py   ← DaVinci Resolve から実行するメインスクリプト
├── config.json             ← ユーザー設定ファイル（ここを編集するだけでOK）
├── requirements.txt        ← pip 依存パッケージ
├── src/
│   ├── davinci_api.py      ← DaVinci Resolve / Fusion API 操作
│   ├── switching_logic.py  ← スイッチング判定ステートマシン
│   ├── shuttle_tracker.py  ← TrackNetV3 推論ラッパー
│   ├── projection.py       ← 360° ↔ 平面 座標変換
│   └── utils.py            ← ロギング・設定読み込み共通処理
├── models/                 ← AIモデルウェイト（自分で配置）
│   ├── TrackNet_best.pt
│   └── InpaintNet_best.pt
└── output/                 ← ログ・CSV 出力先（自動作成）
```

---

## セットアップ手順

### Step 1: Python 環境の準備

DaVinci Resolve は内蔵 Python 3.6 を使用しますが、TrackNet の推論は
**外部 Python 3.10 環境**で行います。

```powershell
# Python 3.10 の仮想環境を作成
python -m venv venv
.\venv\Scripts\activate

# PyTorch（CUDA 11.8 対応版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存パッケージ
pip install -r requirements.txt
```

### Step 2: TrackNetV3 のセットアップ

```powershell
# プロジェクトルートに TrackNetV3 をクローン
git clone https://github.com/qaz812345/TrackNetV3.git

# モデルウェイトをダウンロードして models/ に配置
# （TrackNetV3 の README に記載のダウンロードリンクを参照）
# models/TrackNet_best.pt
# models/InpaintNet_best.pt
```

### Step 3: DJI Reframe プラグインのインストール

1. [DJI 公式サイト](https://www.dji.com/downloads/softwares/dji-reframe-davinci) から DJI Reframe をダウンロード
2. インストーラーを実行（DaVinci Resolve は終了した状態で）
3. DaVinci Resolve 20.0.1 以上が必要

### Step 4: スクリプトを DaVinci Resolve に登録

1. DaVinci Resolve を起動
2. `Preferences > System > General` から `Script console` の場所を確認
3. または `Workspace > Scripts` のフォルダに `badminton_switcher.py` をコピー
   - **Windows**: `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Scripts\`

---

## 使い方

### Phase 1: API 結合テスト（TrackNet 不要）

1. `config.json` を開き、`phase1_dummy_mode` を `true` に設定
2. DaVinci Resolve で 360° 映像クリップをタイムラインに配置
3. Edit ページでそのクリップを選択（クリックでハイライト状態に）
4. `Workspace > Scripts > badminton_switcher` を実行
5. Fusion ページを開いてキーフレームが打刻されているか確認

### Phase 2: シャトル座標単体テスト

```powershell
# venv を有効化した状態で
python src/shuttle_tracker.py \
  --video "testデータ/0719_davinci.mp4" \
  --tracknet models/TrackNet_best.pt \
  --inpaint models/InpaintNet_best.pt \
  --output output/shuttle_coords.csv \
  --debug
```

出力された `output/shuttle_coords.csv` を Excel 等で開き、
シャトル座標が妥当に検出できているか確認してください。

### Phase 3: 本番実行

1. `config.json` で `phase1_dummy_mode` を `false` に変更
2. DJI Reframe プラグインをインストール済みであることを確認
3. DaVinci Resolve で `0719_davinci.mp4` をタイムラインに配置・選択
4. `Workspace > Scripts > badminton_switcher` を実行

---

## config.json パラメータ解説

```json
{
  "camera_setup": {
    "fx_plugin_name": "DJIReframe",   // 使用プラグイン名
    "fx_plugin_fallback": "PanoMap",  // フォールバック（Phase 1 テスト用）
    "yaw_angle_a": -45.0,             // コート左側の Yaw 角度
    "yaw_angle_b": 45.0,              // コート右側の Yaw 角度
    "default_pitch": 0.0,             // カメラの垂直角度（固定）
    "default_fov": 90.0               // 画角（固定）
  },
  "algorithm_tuning": {
    "deadzone_min": 0.45,             // 不感帯の左端（0=左端）
    "deadzone_max": 0.55,             // 不感帯の右端（1=右端）
    "min_hold_frames": 15,            // 切り替え後のクールダウンフレーム数
    "shuttle_confidence_threshold": 0.60,  // この確信度未満はロスト扱い
    "lost_tolerance_frames": 10       // ロスト耐性フレーム数
  },
  "execution_options": {
    "apply_step_keyframes": true,         // ステップキーフレーム打刻を行うか
    "debug_mode": false,                  // DEBUG ログを出力するか
    "phase1_dummy_mode": false,           // true=ダミーモード, false=本番
    "phase1_dummy_switches": [100, 250, 400, 600, 800]  // ダミー切り替えフレーム
  }
}
```

### パラメータ調整のコツ

| 症状 | 調整方法 |
|------|---------|
| 切り替えが頻繁すぎる（チャタリング） | `min_hold_frames` を増やす（例: 15→30） |
| ネット際で不必要に切り替わる | `deadzone_min`〜`deadzone_max` を広げる（例: 0.40〜0.60） |
| 検出が少なくカクカクする | `shuttle_confidence_threshold` を下げる（例: 0.60→0.45） |
| コートの向きに対して画角がズレる | `yaw_angle_a` / `yaw_angle_b` を調整 |

---

## トラブルシューティング

### 「DaVinci Resolve API に接続できません」
→ DaVinci Resolve が起動中か確認。`Workspace > Scripts` から実行しているか確認。

### 「クリップが選択されていません」
→ Edit ページで 360° クリップをクリックしてからスクリプトを実行。

### 「ノード 'DJIReframe' の作成に失敗しました」
→ DJI Reframe プラグインがインストールされているか確認。
   または `config.json` の `phase1_dummy_mode: true` で PanoMap を使ってテスト。

### 「TrackNetV3 が見つかりません」
→ `git clone` したディレクトリがプロジェクトルートに存在するか確認。
   または `PYTHONPATH` に TrackNetV3 のディレクトリを追加。

---

## ライセンス・利用規約

- 本スクリプトは個人・商用問わず自由に使用可能です
- TrackNetV3 のライセンスは [公式リポジトリ](https://github.com/qaz812345/TrackNetV3) を参照
- DJI Reframe のライセンスは DJI の利用規約に従います
