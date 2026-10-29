"""
shuttle_tracker.py - TrackNetV3 を使ったシャトル追跡ラッパー

設計書 §2「レイヤー1: 映像解析」の実装。
TrackNetV3 を使い、各フレームのシャトル座標 (x_normalized, confidence) を抽出する。

依存:
  - TrackNetV3 リポジトリを git clone し、PYTHONPATH に追加しておく必要あり
  - https://github.com/qaz812345/TrackNetV3

使い方（CLI単体テスト）:
  python src/shuttle_tracker.py \
    --video testデータ/0719_davinci.mp4 \
    --tracknet models/TrackNet_best.pt \
    --debug
"""

from __future__ import annotations
import sys
import os
import csv
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger("BadmintonSwitcher")


# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

# TrackNet 入力サイズ（学習時の設定に合わせる）
TRACKNET_INPUT_W = 512
TRACKNET_INPUT_H = 288

# ヒートマップ閾値
DEFAULT_HEATMAP_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 前処理・後処理
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frames(
    frames: List[np.ndarray],
    target_size: Tuple[int, int] = (TRACKNET_INPUT_W, TRACKNET_INPUT_H),
):
    """
    連続する3フレームをリサイズ・正規化し、PyTorch テンソルに変換する。

    Args:
        frames:      連続する3フレーム [f_prev, f_curr, f_next] の BGR NumPy 配列
        target_size: モデル入力サイズ (width, height)

    Returns:
        torch.Tensor: shape (1, 9, H, W) の float32 テンソル
    """
    import torch  # 遅延インポート（DaVinci 環境でも import エラーを出さないため）

    resized = [cv2.resize(f, target_size) for f in frames]
    # BGR → RGB
    rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized]
    # 正規化 (0.0〜1.0) & チャネルファースト (H,W,C) → (C,H,W)
    norm_list = [np.transpose(f.astype(np.float32) / 255.0, (2, 0, 1)) for f in rgb_list]
    # 3フレームをチャネル方向に結合 → (9, H, W)
    stacked = np.concatenate(norm_list, axis=0)
    # バッチ次元追加 → (1, 9, H, W)
    return torch.from_numpy(stacked).unsqueeze(0)


def extract_shuttle_coordinate(
    heatmap: np.ndarray,
    orig_shape: Tuple[int, int],
    threshold: float = DEFAULT_HEATMAP_THRESHOLD,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    TrackNet の出力ヒートマップからシャトルの重心座標を算出する。

    Args:
        heatmap:    TrackNet 出力ヒートマップ (H', W') の float32 配列
        orig_shape: 元フレームの (height, width)
        threshold:  ヒートマップの二値化閾値

    Returns:
        ((x, y), confidence): 検出座標と確信度。未検出時は (None, 0.0)
    """
    orig_h, orig_w = orig_shape
    map_h, map_w = heatmap.shape

    # 閾値処理
    binary = (heatmap > threshold).astype(np.uint8)

    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if num_labels <= 1:
        # 背景のみ → シャトル未検出
        return None, 0.0

    # 最大面積の連結成分（背景=0 を除外）
    max_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx_map, cy_map = centroids[max_idx]

    # 元の解像度にスケーリング
    x_real = int(cx_map * (orig_w / map_w))
    y_real = int(cy_map * (orig_h / map_h))
    confidence = float(np.max(heatmap[labels == max_idx]))

    return (x_real, y_real), confidence


# ─────────────────────────────────────────────────────────────────────────────
# ShuttleTracker クラス
# ─────────────────────────────────────────────────────────────────────────────

class ShuttleTracker:
    """
    TrackNetV3 を使ったシャトル追跡クラス。

    使用例:
        tracker = ShuttleTracker(
            tracknet_path="models/TrackNet_best.pt",
            inpaintnet_path="models/InpaintNet_best.pt",
        )
        results = tracker.analyze_video("testデータ/0719_davinci.mp4")
        # results: List[(frame_idx, (x, y) or None, confidence)]
    """

    def __init__(
        self,
        tracknet_path: str,
        inpaintnet_path: Optional[str] = None,
        device: str = "auto",
        input_size: Tuple[int, int] = (TRACKNET_INPUT_W, TRACKNET_INPUT_H),
        heatmap_threshold: float = DEFAULT_HEATMAP_THRESHOLD,
        use_inpaintnet: bool = True,
        court_yaw: float = 0.0,
        court_pitch: float = -10.0,
        court_fov: float = 100.0,
        gnomonic_out_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Args:
            tracknet_path:     TrackNet モデルウェイトのパス (.pt)
            inpaintnet_path:   InpaintNet モデルウェイトのパス (.pt)。None で無効。
            device:            "auto" / "cuda" / "cpu"
            input_size:        TrackNet 入力サイズ (width, height)
            heatmap_threshold: ヒートマップ閾値
            use_inpaintnet:    InpaintNet による軌跡補完を行うか
            court_yaw:         コート中央方向の Yaw（ERP→Gnomonic 変換用）
            court_pitch:       コート中央方向の Pitch
            court_fov:         Gnomonic 切り出しの FOV
            gnomonic_out_size: Gnomonic 出力サイズ (width, height)
        """
        import torch

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"デバイス: {self.device}")

        # TrackNet モデルロード
        self.tracknet = self._load_tracknet(tracknet_path)

        # InpaintNet モデルロード（オプション）
        self.inpaintnet = None
        if use_inpaintnet and inpaintnet_path:
            self.inpaintnet = self._load_inpaintnet(inpaintnet_path)

        self.input_size = input_size
        self.heatmap_threshold = heatmap_threshold

        # Gnomonic 変換パラメータ
        self.court_yaw = court_yaw
        self.court_pitch = court_pitch
        self.court_fov = court_fov
        self.gnomonic_out_size = gnomonic_out_size

    def _load_tracknet(self, model_path: str):
        """TrackNet モデルをロードする"""
        import torch
        try:
            from TrackNetV3.utils.general import get_model  # type: ignore
        except ImportError:
            try:
                from utils.general import get_model  # type: ignore
            except ImportError:
                raise ImportError(
                    "TrackNetV3 が見つかりません。\n"
                    "README の指示に従い TrackNetV3 をセットアップしてください。\n"
                    "https://github.com/qaz812345/TrackNetV3"
                )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        seq_len = checkpoint['param_dict']['seq_len']
        bg_mode = checkpoint['param_dict']['bg_mode']
        
        model = get_model('TrackNet', seq_len=seq_len, bg_mode=bg_mode)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        logger.info(f"TrackNet ロード完了: {model_path} (seq_len={seq_len}, bg_mode={bg_mode})")
        
        self.seq_len = seq_len
        self.bg_mode = bg_mode
        return model

    def _load_inpaintnet(self, model_path: str):
        """InpaintNet モデルをロードする"""
        import torch
        try:
            from TrackNetV3.utils.general import get_model  # type: ignore
        except ImportError:
            try:
                from utils.general import get_model  # type: ignore
            except ImportError:
                logger.warning("InpaintNet が見つかりません。軌跡補完を無効化します。")
                return None

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model = get_model('InpaintNet')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        logger.info(f"InpaintNet ロード完了: {model_path}")
        return model

    def analyze_video(
        self,
        video_path: str,
        progress_interval: int = 300,
        tmp_flat_path: str = "output/tmp_flat.mp4",
    ) -> List[Tuple[int, Optional[Tuple[int, int]], float]]:
        """
        動画全フレームを解析し、各フレームのシャトル座標を返す。

        内部処理:
          1. ERP動画 → Gnomonic平面動画に変換（一時ファイル）
          2. TrackNetV3 公式の Shuttlecock_Trajectory_Dataset で推論
          3. 平面座標 → ERP正規化X座標 に逆変換

        Returns:
            List of (frame_idx, coord_or_None, confidence)
        """
        import torch
        import sys as _sys
        from pathlib import Path as _Path

        # TrackNetV3 を sys.path に追加
        tracknetv3_dir = str(_Path(__file__).parent.parent / "TrackNetV3")
        if tracknetv3_dir not in _sys.path:
            _sys.path.insert(0, tracknetv3_dir)

        from utils.general import get_model, HEIGHT, WIDTH  # type: ignore
        from dataset import Shuttlecock_Trajectory_Dataset   # type: ignore
        from torch.utils.data import DataLoader
        from src.projection import extract_court_region, gnomonic_coord_to_erp_x_normalized

        # ── Step 1: 元動画の情報を取得 ──────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画ファイルを開けません: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"動画解析開始: {Path(video_path).name} ({orig_w}x{orig_h}, {fps:.1f}fps, {total_frames}f)")

        # ── Step 2: Gnomonic 平面動画を一時ファイルに書き出し ────────────
        tmp_flat_path = str(Path(tmp_flat_path).resolve())
        Path(tmp_flat_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        gno_w, gno_h = self.gnomonic_out_size
        writer = cv2.VideoWriter(tmp_flat_path, fourcc, fps, (gno_w, gno_h))

        logger.info(f"Gnomonic変換中... → {tmp_flat_path}")
        actual_total = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flat = extract_court_region(
                frame,
                court_yaw=self.court_yaw,
                court_pitch=self.court_pitch,
                fov_deg=self.court_fov,
                out_size=self.gnomonic_out_size,
            )
            writer.write(flat)
            actual_total += 1
            if actual_total % 500 == 0:
                logger.info(f"  変換中: {actual_total}f 完了...")
        cap.release()
        writer.release()
        logger.info(f"Gnomonic変換完了: {actual_total} フレーム")

        # ── Step 3: TrackNetV3 公式ロジックで推論 ─────────────────────
        seq_len = self.seq_len
        bg_mode = self.bg_mode

        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode='heatmap',
            bg_mode=bg_mode,
            frame_arr=self._load_flat_frames(tmp_flat_path),
        )
        num_workers = 0  # Windowsではnum_workers=0推奨
        data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                                 num_workers=num_workers, drop_last=False)

        # 重み付きアンサンブル（公式"weight"モード簡易版）
        from test import predict_location, get_ensemble_weight, generate_inpaint_mask  # type: ignore
        weight = get_ensemble_weight(seq_len, 'weight')

        video_len = actual_total
        num_sample = video_len - seq_len + 1
        sample_count = 0
        buffer_size = seq_len - 1
        batch_i  = torch.arange(seq_len)
        frame_i  = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)

        pred_dict: dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

        img_scaler = (orig_w / WIDTH, orig_h / HEIGHT)

        self.tracknet.eval()
        with torch.no_grad():
            for step, (i_idx, x) in enumerate(data_loader):
                x = x.float().to(self.device)
                b_size = i_idx.shape[0]
                y_pred = self.tracknet(x).detach().cpu()

                y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                ensemble_i      = torch.empty((0, 1, 2),          dtype=torch.float32)
                ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        y_ens = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        y_ens = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                    ensemble_i      = torch.cat((ensemble_i,      i_idx[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                        y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                        for f in range(1, seq_len):
                            y_ens = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i      = torch.cat((ensemble_i,      i_idx[-1][f].reshape(1, 1, 2)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

                # バッファ先頭を捨てる
                y_pred_buffer = y_pred_buffer[b_size:]

                # 座標抽出
                ensemble_y_pred_bin = ensemble_y_pred > 0.5
                for b in range(ensemble_i.shape[0]):
                    f_i = int(ensemble_i[b][0][1].item())
                    hm  = ensemble_y_pred_bin[b][0].numpy().astype(np.uint8)
                    bbox = predict_location(hm)
                    cx  = int(bbox[0] + bbox[2] / 2)
                    cy  = int(bbox[1] + bbox[3] / 2)
                    vis = 0 if cx == 0 and cy == 0 else 1
                    pred_dict['Frame'].append(f_i)
                    pred_dict['X'].append(cx)
                    pred_dict['Y'].append(cy)
                    pred_dict['Visibility'].append(vis)

                if (step + 1) % 10 == 0:
                    logger.info(f"  推論進捗: step {step+1}")

        logger.info(f"TrackNet推論完了")

        # ── Step 4: 平面座標 → ERP正規化X座標 に逆変換 ─────────────────
        results: List[Tuple[int, Optional[Tuple[int, int]], float]] = []
        frame_to_pred = {pred_dict['Frame'][i]: i for i in range(len(pred_dict['Frame']))}

        detected_count = 0
        for frame_idx in range(actual_total):
            if frame_idx not in frame_to_pred:
                results.append((frame_idx, None, 0.0))
                continue

            idx = frame_to_pred[frame_idx]
            vis = pred_dict['Visibility'][idx]
            cx_flat = pred_dict['X'][idx]
            cy_flat = pred_dict['Y'][idx]

            if vis == 0 or (cx_flat == 0 and cy_flat == 0):
                results.append((frame_idx, None, 0.0))
                continue

            # 平面画像上の座標 → 元のGnomonic解像度にスケール
            cx_gno = int(cx_flat * img_scaler[0])
            cy_gno = int(cy_flat * img_scaler[1])

            x_erp_norm = gnomonic_coord_to_erp_x_normalized(
                cx=cx_gno, cy=cy_gno,
                out_width=gno_w, out_height=gno_h,
                yaw_deg=self.court_yaw, pitch_deg=self.court_pitch,
                fov_deg=self.court_fov, erp_width=orig_w,
            )
            x_erp_px = int(x_erp_norm * orig_w)
            results.append((frame_idx, (x_erp_px, cy_gno), 1.0))
            detected_count += 1

        logger.info(f"解析完了: 検出率 {detected_count}/{actual_total} ({detected_count/actual_total:.1%})")
        return results

    def _load_flat_frames(self, video_path: str) -> np.ndarray:
        """一時Gnomonic動画を読み込んでnumpy配列として返す"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame[:, :, ::-1])  # BGR→RGB
        cap.release()
        return np.array(frames)


# ─────────────────────────────────────────────────────────────────────────────
# CSV 出力
# ─────────────────────────────────────────────────────────────────────────────

def save_coords_to_csv(
    results: List[Tuple[int, Optional[Tuple[int, int]], float]],
    output_path: str,
    fps: float = 30.0,
):
    """
    解析結果を CSV ファイルに保存する。

    CSV 列: frame, timecode, x_pixel, y_pixel, confidence, detected
    """
    from src.utils import frames_to_timecode

    Path(output_path).parent.mkdir(exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timecode", "x_pixel", "y_pixel", "confidence", "detected"])
        for frame_idx, coord, conf in results:
            tc = frames_to_timecode(frame_idx, fps)
            if coord is not None:
                x, y = coord
                writer.writerow([frame_idx, tc, x, y, f"{conf:.4f}", 1])
            else:
                writer.writerow([frame_idx, tc, "", "", "0.0000", 0])

    logger.info(f"CSV 保存完了: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 単体テスト（Phase 2 デバッグ用）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrackNetV3 によるシャトル追跡 CLI テスト"
    )
    parser.add_argument("--video",     required=True, help="入力動画パス")
    parser.add_argument("--tracknet",  required=True, help="TrackNet モデルウェイトパス")
    parser.add_argument("--inpaint",   default=None,  help="InpaintNet モデルウェイトパス")
    parser.add_argument("--output",    default="output/shuttle_coords.csv", help="CSV 出力パス")
    parser.add_argument("--device",    default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug",     action="store_true", help="DEBUG ログを有効化")
    args = parser.parse_args()

    # プロジェクトルートを sys.path に追加
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils import setup_logger
    setup_logger(debug=args.debug)

    cap_tmp = cv2.VideoCapture(args.video)
    fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    cap_tmp.release()

    tracker = ShuttleTracker(
        tracknet_path=args.tracknet,
        inpaintnet_path=args.inpaint,
        device=args.device,
    )
    results = tracker.analyze_video(args.video)
    save_coords_to_csv(results, args.output, fps=fps)

    print(f"\n✅ 解析完了。CSV を確認してください: {args.output}")
