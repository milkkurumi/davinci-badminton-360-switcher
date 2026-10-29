"""
utils.py - ロギング・共通ユーティリティ
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# ロガーセットアップ
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "BadmintonSwitcher", debug: bool = False) -> logging.Logger:
    """
    コンソールとファイルの両方に出力するロガーを作成・返す。

    Args:
        name:  ロガー名
        debug: True の場合 DEBUG レベル、False の場合 INFO レベル

    Returns:
        設定済み logging.Logger オブジェクト
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # 既にハンドラが設定されている場合は再設定しない
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # コンソールハンドラ
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ファイルハンドラ（output/ ディレクトリに保存）
    try:
        log_dir = Path(__file__).parent.parent / "output"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"run_{timestamp}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # ファイルログ失敗はサイレント無視

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 設定ファイルローダー
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """
    config.json を読み込んで辞書として返す。

    Args:
        config_path: config.json の絶対パス。None の場合スクリプトの隣を自動検索。

    Returns:
        設定辞書

    Raises:
        FileNotFoundError: config.json が見つからない場合
        json.JSONDecodeError: JSON フォーマットが不正な場合
    """
    if config_path is None:
        # このファイル (src/utils.py) の親ディレクトリ（プロジェクトルート）を探す
        config_path = Path(__file__).parent.parent / "config.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


# ─────────────────────────────────────────────────────────────────────────────
# 汎用ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def normalize_x(x_pixel: int, frame_width: int) -> float:
    """
    ピクセルX座標を 0.0（左端）〜 1.0（右端）の正規化値に変換する。

    Args:
        x_pixel:     ピクセル単位のX座標
        frame_width: フレームの横幅（ピクセル）

    Returns:
        正規化されたX座標 (0.0〜1.0)
    """
    if frame_width <= 0:
        return 0.5
    return max(0.0, min(1.0, x_pixel / frame_width))


def frames_to_timecode(frame_idx: int, fps: float) -> str:
    """
    フレーム番号をタイムコード文字列（HH:MM:SS:FF）に変換する。

    Args:
        frame_idx: フレーム番号（0始まり）
        fps:       フレームレート

    Returns:
        タイムコード文字列
    """
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int(frame_idx % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def ensure_output_dir(base_dir: str = None) -> Path:
    """
    output/ ディレクトリが存在しない場合は作成して Path を返す。

    Args:
        base_dir: プロジェクトルートのパス。None の場合自動検出。

    Returns:
        output ディレクトリの Path オブジェクト
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    output_dir = Path(base_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
