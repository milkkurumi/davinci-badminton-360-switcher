"""
switching_logic.py - スイッチング判定ステートマシン

設計書 §3 「コア・アルゴリズム（スイッチングロジック設計）」の実装。
3つの制御ルール:
  1. 不感帯（デッドゾーン）: ネット中央付近では判定しない
  2. 最低ホールドフレーム（クールダウン）: 切り替え直後はロック
  3. ロスト時のホールド: 検出失敗時は直前状態を維持
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("BadmintonSwitcher")


# ─────────────────────────────────────────────────────────────────────────────
# データクラス
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwitchingResult:
    """1フレームの判定結果"""
    frame_idx: int
    state: str                          # "A" or "B"
    x_normalized: Optional[float]       # 0.0〜1.0 (None=ロスト)
    confidence: float
    switched: bool                      # このフレームで切り替わったか
    reason: str                         # デバッグ用: 判定理由


@dataclass
class SwitchingStats:
    """処理全体の統計情報"""
    total_frames: int = 0
    detected_frames: int = 0
    lost_frames: int = 0
    deadzone_frames: int = 0
    cooldown_frames: int = 0
    total_switches: int = 0
    state_a_frames: int = 0
    state_b_frames: int = 0

    @property
    def detection_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.detected_frames / self.total_frames

    def summary(self) -> str:
        return (
            f"総フレーム: {self.total_frames} | "
            f"検出率: {self.detection_rate:.1%} | "
            f"切り替え回数: {self.total_switches} | "
            f"STATE_A: {self.state_a_frames}f / STATE_B: {self.state_b_frames}f"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ステートマシン本体
# ─────────────────────────────────────────────────────────────────────────────

class SwitchingStateMachine:
    """
    バドミントン画角自動切り替えステートマシン。

    STATE_A: コート左側（または手前）を捉える固定画角 (yaw_angle_a)
    STATE_B: コート右側（または奥）を捉える固定画角 (yaw_angle_b)

    判定ルール（優先順位順）:
      1. 確信度 < threshold → ロスト扱い、直前状態を維持
      2. クールダウン中 → 切り替え判定をスキップ
      3. 不感帯内 → 切り替え判定をスキップ
      4. x < deadzone_min → STATE_A へ遷移
      5. x > deadzone_max → STATE_B へ遷移
    """

    STATE_A = "A"
    STATE_B = "B"

    def __init__(
        self,
        yaw_angle_a: float = -45.0,
        yaw_angle_b: float = 45.0,
        deadzone_min: float = 0.45,
        deadzone_max: float = 0.55,
        min_hold_frames: int = 15,
        confidence_threshold: float = 0.60,
        lost_tolerance_frames: int = 10,
        initial_state: str = STATE_A,
    ):
        """
        Args:
            yaw_angle_a:           STATE_A の Yaw 角度（度）
            yaw_angle_b:           STATE_B の Yaw 角度（度）
            deadzone_min:          不感帯の左端 (0.0〜1.0)
            deadzone_max:          不感帯の右端 (0.0〜1.0)
            min_hold_frames:       切り替え後のクールダウンフレーム数
            confidence_threshold:  この値未満の確信度はロスト扱い
            lost_tolerance_frames: ロストが何フレーム続いてもOKか（将来拡張用）
            initial_state:         初期状態 ("A" or "B")
        """
        self.yaw_angle_a = yaw_angle_a
        self.yaw_angle_b = yaw_angle_b
        self.deadzone_min = deadzone_min
        self.deadzone_max = deadzone_max
        self.min_hold_frames = min_hold_frames
        self.confidence_threshold = confidence_threshold
        self.lost_tolerance_frames = lost_tolerance_frames

        self.current_state: str = initial_state
        self.last_switch_frame: int = -min_hold_frames  # 初回は即判定OK
        self.consecutive_lost: int = 0

        # 切り替えポイントの記録: { frame_N: new_yaw_value }
        self._switching_points: Dict[int, float] = {}
        # 全フレームの判定ログ
        self._frame_log: list[SwitchingResult] = []
        # 統計
        self.stats = SwitchingStats()

    # ─── パブリックAPI ────────────────────────────────────────────────────

    def process(
        self,
        frame_idx: int,
        x_normalized: Optional[float],
        confidence: float,
    ) -> SwitchingResult:
        """
        1フレーム分の判定を実行する。

        Args:
            frame_idx:    フレーム番号（0始まり）
            x_normalized: シャトルの正規化X座標 (0.0〜1.0)。ロスト時は None。
            confidence:   検出確信度 (0.0〜1.0)

        Returns:
            SwitchingResult: 判定結果
        """
        self.stats.total_frames += 1
        switched = False
        reason = ""

        # ─── ルール1: ロスト判定 ───────────────────────────────────────
        is_lost = (x_normalized is None) or (confidence < self.confidence_threshold)

        if is_lost:
            self.consecutive_lost += 1
            self.stats.lost_frames += 1
            reason = f"ロスト (conf={confidence:.2f})"
            result = SwitchingResult(
                frame_idx=frame_idx,
                state=self.current_state,
                x_normalized=x_normalized,
                confidence=confidence,
                switched=False,
                reason=reason,
            )
            self._frame_log.append(result)
            self._update_state_stats()
            return result

        self.consecutive_lost = 0
        self.stats.detected_frames += 1

        # ─── ルール2: クールダウン中 ──────────────────────────────────
        frames_since_switch = frame_idx - self.last_switch_frame
        if frames_since_switch < self.min_hold_frames:
            reason = f"クールダウン中 ({frames_since_switch}/{self.min_hold_frames}f)"
            self.stats.cooldown_frames += 1
            result = SwitchingResult(
                frame_idx=frame_idx,
                state=self.current_state,
                x_normalized=x_normalized,
                confidence=confidence,
                switched=False,
                reason=reason,
            )
            self._frame_log.append(result)
            self._update_state_stats()
            return result

        # ─── ルール3: 不感帯 ──────────────────────────────────────────
        if self.deadzone_min <= x_normalized <= self.deadzone_max:
            reason = f"不感帯内 (x={x_normalized:.3f})"
            self.stats.deadzone_frames += 1
            result = SwitchingResult(
                frame_idx=frame_idx,
                state=self.current_state,
                x_normalized=x_normalized,
                confidence=confidence,
                switched=False,
                reason=reason,
            )
            self._frame_log.append(result)
            self._update_state_stats()
            return result

        # ─── ルール4〜5: 状態遷移判定 ────────────────────────────────
        target_state = self.current_state
        if x_normalized < self.deadzone_min:
            target_state = self.STATE_A
        elif x_normalized > self.deadzone_max:
            target_state = self.STATE_B

        if target_state != self.current_state:
            # 切り替え実行
            old_state = self.current_state
            self.current_state = target_state
            self.last_switch_frame = frame_idx
            self.stats.total_switches += 1
            new_yaw = self.yaw_angle_a if target_state == self.STATE_A else self.yaw_angle_b
            self._switching_points[frame_idx] = new_yaw
            switched = True
            reason = (
                f"切り替え: STATE_{old_state}→STATE_{target_state} "
                f"(x={x_normalized:.3f}, yaw={new_yaw:+.1f}°)"
            )
            logger.debug(f"  [Frame {frame_idx:6d}] {reason}")
        else:
            reason = f"維持: STATE_{self.current_state} (x={x_normalized:.3f})"

        result = SwitchingResult(
            frame_idx=frame_idx,
            state=self.current_state,
            x_normalized=x_normalized,
            confidence=confidence,
            switched=switched,
            reason=reason,
        )
        self._frame_log.append(result)
        self._update_state_stats()
        return result

    def get_switching_points(self) -> Dict[int, float]:
        """
        切り替えポイント辞書を返す。
        形式: { frame_N: new_yaw_value }
        DaVinci API 連携レイヤーに渡す。
        """
        return dict(sorted(self._switching_points.items()))

    def get_initial_yaw(self) -> float:
        """初期状態の Yaw 角度を返す"""
        return self.yaw_angle_a if self.current_state == self.STATE_A else self.yaw_angle_b

    def get_frame_log(self) -> list:
        """全フレームの判定ログを返す（デバッグ・CSV出力用）"""
        return self._frame_log

    def reset(self):
        """ステートマシンを初期状態にリセットする"""
        self.current_state = self.STATE_A
        self.last_switch_frame = -self.min_hold_frames
        self.consecutive_lost = 0
        self._switching_points = {}
        self._frame_log = []
        self.stats = SwitchingStats()

    # ─── 内部ヘルパー ─────────────────────────────────────────────────────

    def _update_state_stats(self):
        if self.current_state == self.STATE_A:
            self.stats.state_a_frames += 1
        else:
            self.stats.state_b_frames += 1


# ─────────────────────────────────────────────────────────────────────────────
# ダミーデータ生成（Phase 1 テスト用）
# ─────────────────────────────────────────────────────────────────────────────

def generate_dummy_coords(
    total_frames: int,
    switch_frames: list[int],
) -> list[Tuple[int, Optional[float], float]]:
    """
    Phase 1 API 結合テスト用のダミーシャトル座標を生成する。

    Args:
        total_frames:  総フレーム数
        switch_frames: 画角が切り替わるべきフレーム番号のリスト

    Returns:
        List of (frame_idx, x_normalized, confidence)
    """
    coords = []
    current_side = 0.2  # 左側（STATE_A 側）から開始

    switch_set = set(switch_frames)
    sides = [0.2, 0.8]  # 左・右の代表座標
    side_idx = 0

    for f in range(total_frames):
        if f in switch_set:
            side_idx = 1 - side_idx  # 左右を反転
        x = sides[side_idx]
        coords.append((f, x, 0.90))

    return coords
