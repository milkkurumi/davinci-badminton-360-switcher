"""
projection.py - Equirectangular ↔ Gnomonic（透視投影）変換

TrackNetV3 は平面画像を前提に学習されているため、
360°映像（Equirectangular）をそのまま入力すると精度が大幅低下する。
このモジュールでバドミントンコート領域を歪みのない平面に切り出してから推論する。

座標変換の流れ:
  ERP画像 → Gnomonic切り出し（TrackNetへ入力）
  TrackNet出力(x,y) → 球面座標(θ,φ) → ERP座標(X,Y) → 正規化X (0.0〜1.0)
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# ERP → Gnomonic 切り出し変換
# ─────────────────────────────────────────────────────────────────────────────

def equirect_to_gnomonic(
    erp_frame: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_width: int = 1280,
    out_height: int = 720,
) -> np.ndarray:
    """
    Equirectangular フレームから指定視点・視野角の平面画像（Rectilinear）を切り出す。

    Args:
        erp_frame:  ERP 画像 (H, W, 3) の NumPy 配列
        yaw_deg:    水平回転角（度）。0=正面、正=右
        pitch_deg:  垂直傾き角（度）。0=水平、正=上
        fov_deg:    垂直方向の視野角（度）
        out_width:  出力画像の幅（ピクセル）
        out_height: 出力画像の高さ（ピクセル）

    Returns:
        切り出した平面画像 (out_height, out_width, 3) の NumPy 配列
    """
    erp_h, erp_w = erp_frame.shape[:2]

    # ラジアン変換
    yaw_rad   = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    fov_rad   = math.radians(fov_deg)

    # 出力画像の各ピクセルに対応する正規化座標 (-1 〜 +1)
    f = np.float32(1.0 / math.tan(fov_rad / 2.0))
    xs = np.linspace(-(out_width / 2.0), out_width / 2.0, out_width,  dtype=np.float32) / (out_height / 2.0)
    ys = np.linspace(-(out_height / 2.0), out_height / 2.0, out_height, dtype=np.float32) / (out_height / 2.0)
    xv, yv = np.meshgrid(xs, ys)
    yv = -yv  # 画像のY軸（下が正）を数学のY軸（上が正）に変換

    # 正規化 3D 方向ベクトル（カメラ座標系）
    zv = np.full_like(xv, f)
    norm = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2)
    xv /= norm
    yv /= norm
    zv /= norm

    # pitch（X軸回転）適用
    cp = np.float32(math.cos(-pitch_rad))
    sp = np.float32(math.sin(-pitch_rad))
    xv2 =  xv
    yv2 =  cp * yv - sp * zv
    zv2 =  sp * yv + cp * zv

    # yaw（Y軸回転）適用
    cy = np.float32(math.cos(yaw_rad))
    sy = np.float32(math.sin(yaw_rad))
    xv3 = cy * xv2 + sy * zv2
    yv3 = yv2
    zv3 = -sy * xv2 + cy * zv2

    # 球面座標へ変換
    lon = np.arctan2(xv3, zv3)          # -π 〜 +π
    lat = np.arcsin(np.clip(yv3, -1, 1))  # -π/2 〜 +π/2

    # ERP 画像のピクセル座標へマッピング
    map_x = ((lon / math.pi + 1.0) / 2.0 * erp_w).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * erp_h).astype(np.float32)

    # バイリニア補間でサンプリング
    out = cv2.remap(
        erp_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,  # 左右端をシームレスに折り返す
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gnomonic 座標 → ERP 正規化 X 座標 への逆変換
# ─────────────────────────────────────────────────────────────────────────────

def gnomonic_coord_to_erp_x_normalized(
    cx: int,
    cy: int,
    out_width: int,
    out_height: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    erp_width: int,
) -> float:
    """
    Gnomonic 切り出し画像上の検出座標を ERP 映像上の正規化 X 座標 (0.0〜1.0) に変換する。

    Args:
        cx, cy:      Gnomonic 画像上のシャトル検出座標（ピクセル）
        out_width:   Gnomonic 画像の幅
        out_height:  Gnomonic 画像の高さ
        yaw_deg:     切り出し時の Yaw 角度
        pitch_deg:   切り出し時の Pitch 角度
        fov_deg:     切り出し時の FOV
        erp_width:   元の ERP フレームの幅

    Returns:
        ERP フレーム上の正規化 X 座標 (0.0〜1.0)
    """
    fov_rad   = math.radians(fov_deg)
    yaw_rad   = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)

    # Gnomonic 画像座標を正規化方向ベクトルへ逆変換
    f = 1.0 / math.tan(fov_rad / 2.0)
    x_norm = (cx - out_width  / 2.0) / (out_height / 2.0)
    y_norm = (cy - out_height / 2.0) / (out_height / 2.0)
    z_norm = f

    norm = math.sqrt(x_norm ** 2 + y_norm ** 2 + z_norm ** 2)
    x_norm /= norm
    y_norm /= norm
    z_norm /= norm

    # pitch 逆回転
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    x2 =  x_norm
    y2 =  cp * y_norm - sp * z_norm
    z2 =  sp * y_norm + cp * z_norm

    # yaw 逆回転
    cy_, sy_ = math.cos(-yaw_rad), math.sin(-yaw_rad)
    x3 = cy_ * x2 + sy_ * z2
    z3 = -sy_ * x2 + cy_ * z2

    # 経度（lon）を算出
    lon = math.atan2(x3, z3)  # -π 〜 +π

    # ERP 上の正規化 X 座標に変換
    x_erp_normalized = (lon / math.pi + 1.0) / 2.0  # 0.0〜1.0

    return max(0.0, min(1.0, x_erp_normalized))


# ─────────────────────────────────────────────────────────────────────────────
# コンビニエンス関数
# ─────────────────────────────────────────────────────────────────────────────

def extract_court_region(
    erp_frame: np.ndarray,
    court_yaw: float = 0.0,
    court_pitch: float = -10.0,
    fov_deg: float = 100.0,
    out_size: Tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """
    バドミントンコート全体を含む平面領域を ERP 映像から切り出す。

    ネット中央固定カメラ想定のデフォルト値:
      - Yaw  = 0.0°  (正面)
      - Pitch = -10.0° (コートが水平より少し下にある想定)
      - FOV  = 100.0° (コート全体が収まる広角)

    Args:
        erp_frame:   ERP 映像フレーム
        court_yaw:   コート中央方向の Yaw（度）
        court_pitch: コート中央方向の Pitch（度）
        fov_deg:     視野角（度）
        out_size:    出力サイズ (width, height)

    Returns:
        切り出した平面コート映像
    """
    out_w, out_h = out_size
    return equirect_to_gnomonic(
        erp_frame,
        yaw_deg=court_yaw,
        pitch_deg=court_pitch,
        fov_deg=fov_deg,
        out_width=out_w,
        out_height=out_h,
    )
