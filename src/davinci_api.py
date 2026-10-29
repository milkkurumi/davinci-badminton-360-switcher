"""
davinci_api.py - DaVinci Resolve / Fusion Scripting API 操作レイヤー

【重要】DaVinci Resolve の Scripts メニューから実行した場合、
以下のオブジェクトがスクリプトのグローバル変数に自動注入される:
  resolve, fusion, comp, project, timeline, fu, app

そのため bmd.scriptapp("Resolve") は使用せず、
グローバル変数を呼び出し元から受け取る設計にする。
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("BadmintonSwitcher")


# ─────────────────────────────────────────────────────────────────────────────
# DaVinci Resolve グローバル変数の取得ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def get_resolve_globals(caller_globals: dict) -> dict:
    """
    DaVinci Resolve がスクリプト実行時に注入するグローバル変数を取得する。

    DaVinci の Scripts メニューから実行した場合、以下が自動注入される:
      resolve  : Resolve アプリケーションオブジェクト
      fusion   : FusionUI オブジェクト
      comp     : 現在の Fusion Composition
      project  : 現在のプロジェクト
      timeline : 現在のタイムライン

    Args:
        caller_globals: 呼び出し元スクリプトの globals() を渡す

    Returns:
        { "resolve": ..., "fusion": ..., "comp": ..., "project": ..., "timeline": ... }

    Raises:
        RuntimeError: resolve が取得できない場合
    """
    dvr = {
        "resolve":  caller_globals.get("resolve"),
        "fusion":   caller_globals.get("fusion"),
        "comp":     caller_globals.get("comp"),
        "project":  caller_globals.get("project"),
        "timeline": caller_globals.get("timeline"),
    }

    # フォールバック: fusion.GetResolve() で resolve を取得
    if dvr["resolve"] is None and dvr["fusion"] is not None:
        try:
            dvr["resolve"] = dvr["fusion"].GetResolve()
        except Exception:
            pass

    if dvr["resolve"] is None:
        raise RuntimeError(
            "DaVinci Resolve API に接続できません。\n"
            "DaVinci Resolve の Edit ページから スクリプトを実行してください。\n"
            "（Workspace > Scripts > Edit > badminton_switcher）"
        )

    logger.info(f"DaVinci Resolve に接続しました: {dvr['resolve']}")
    return dvr


def get_timeline_fps(timeline) -> float:
    """現在のタイムラインの FPS を返す"""
    try:
        fps_str = timeline.GetSetting("timelineFrameRate")
        return float(fps_str)
    except Exception:
        return 30.0


# ─────────────────────────────────────────────────────────────────────────────
# タイムラインクリップ取得
# ─────────────────────────────────────────────────────────────────────────────

def get_target_timeline_item(timeline, track_index: int = 1):
    """
    タイムラインから処理対象のクリップを取得する。

    優先順位:
      1. GetCurrentVideoItem() で選択中のクリップ
      2. トラック1の先頭クリップ（フォールバック）

    Args:
        timeline:    DaVinci Resolve の Timeline オブジェクト
        track_index: 対象ビデオトラック番号（デフォルト: 1）

    Returns:
        TimelineItem オブジェクト

    Raises:
        RuntimeError: クリップが見つからない場合
    """
    # 方法1: 選択中のクリップ
    try:
        item = timeline.GetCurrentVideoItem()
        if item is not None:
            logger.info(f"選択中クリップ: {item.GetName()}")
            return item
    except Exception:
        pass

    # 方法2: トラックの全クリップから先頭を取得
    try:
        items = timeline.GetItemListInTrack("video", track_index)
        if items:
            item = items[0]
            logger.info(f"トラック{track_index} 先頭クリップ: {item.GetName()}")
            return item
    except Exception:
        pass

    raise RuntimeError(
        "処理対象のクリップが見つかりません。\n"
        "Edit ページで360°映像クリップをクリックして選択してから再実行してください。"
    )


def get_clip_info(timeline_item) -> Dict:
    """タイムラインクリップの基本情報を辞書で返す"""
    name     = timeline_item.GetName()
    start    = timeline_item.GetStart()
    end      = timeline_item.GetEnd()
    duration = end - start

    info = {"name": name, "start_frame": start, "end_frame": end, "duration": duration}
    logger.info(f"クリップ: '{name}' start={start} end={end} duration={duration}f")
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Composition 取得
# ─────────────────────────────────────────────────────────────────────────────

def get_fusion_comp(timeline_item, current_comp=None):
    """
    タイムラインクリップの Fusion Composition を取得する。
    常に timeline_item から最新のものを取得するよう優先する。
    """
    comp = timeline_item.GetFusionCompByIndex(1)
    
    if comp is not None:
        logger.info(f"タイムラインから最新の Fusion Comp を取得（ツール数: {len(comp.GetToolList() or {})}）")
    elif current_comp is not None:
        try:
            tools = current_comp.GetToolList()
            if tools is not None:
                comp = current_comp
                logger.info(f"グローバル comp を使用（ツール数: {len(tools)}）")
        except Exception:
            pass

    if comp is None:
        logger.info("Fusion Comp が存在しないため新規作成します")
        comp = timeline_item.AddFusionComp()

    if comp is None:
        raise RuntimeError(
            "Fusion Composition の取得に失敗しました。\n"
            "クリップを Edit ページで右クリック →「Open in Fusion Page」で開いてから再実行してください。"
        )

    logger.info(f"Fusion Comp 取得成功（ツール数: {len(comp.GetToolList() or {})}）")
    return comp


# ─────────────────────────────────────────────────────────────────────────────
# Reframe ノード操作
# ─────────────────────────────────────────────────────────────────────────────

# プラグイン名 → Fusion ノード AddTool ID のマッピング
PLUGIN_NODE_IDS = {
    "DJIReframe": "DJIReframe",
    "PanoMap":    "PanoMap",
    "GoPro":      "GoPro_FX_Reframe",
    "Transform":  "Transform",
}

# プラグインごとの Yaw / Pitch / FOV パラメータ名（優先順位順）
PLUGIN_YAW_PARAMS = {
    "DJIReframe": ["Yaw", "Pan"],
    "PanoMap":    ["Angle", "Yaw", "Pan"],
    "GoPro":      ["Pan", "Yaw"],
    "Transform":  ["Angle"],  # Transformノードの場合は回転角を使用
}
PLUGIN_PITCH_PARAMS = {
    "DJIReframe": ["Pitch", "Tilt"],
    "PanoMap":    ["Tilt", "Pitch"],
    "GoPro":      ["Tilt", "Pitch"],
    "Transform":  ["dummy_pitch"],
}
PLUGIN_FOV_PARAMS = {
    "DJIReframe": ["Fov", "FOV", "FieldOfView"],
    "PanoMap":    ["FOV", "Fov"],
    "GoPro":      ["FieldOfView", "FOV"],
    "Transform":  ["dummy_fov"],  # Sizeに90.0が入って超ズームになるのを防ぐ
}


def _find_param(node, candidates: list) -> Optional[Tuple[str, object]]:
    """
    候補パラメータ名リストから実際に存在するものを返す。

    Returns:
        (param_name, param_object) または None
    """
    for name in candidates:
        p = getattr(node, name, None)
        if p is not None:
            return name, p
    return None


def find_or_create_reframe_node(comp, plugin_name: str = "Transform"):
    """
    Fusion Comp 内で Reframe ノードを検索し、なければ作成して接続する。

    接続構成: MediaIn1 → [ReframeNode] → MediaOut1

    Args:
        comp:        Fusion Composition オブジェクト
        plugin_name: 使用するプラグイン名

    Returns:
        (ofx_node, yaw_param_name, pitch_param_name, fov_param_name)

    Raises:
        RuntimeError: ノードの作成に失敗した場合
    """
    node_id       = PLUGIN_NODE_IDS.get(plugin_name, plugin_name)
    yaw_cands     = PLUGIN_YAW_PARAMS.get(plugin_name, ["Yaw"])
    pitch_cands   = PLUGIN_PITCH_PARAMS.get(plugin_name, ["Pitch"])
    fov_cands     = PLUGIN_FOV_PARAMS.get(plugin_name, ["Fov"])

    # 既存ノードを検索
    existing_node = None
    tool_list = comp.GetToolList() or {}
    for _, tool in tool_list.items():
        try:
            attrs = tool.GetAttrs()
            if attrs.get("TOOLS_RegID", "") == node_id:
                existing_node = tool
                logger.info(f"既存の {node_id} ノードを発見")
                break
        except Exception:
            continue

    if existing_node is None:
        # 新規ノード作成
        logger.info(f"{node_id} ノードを新規作成します")
        existing_node = comp.AddTool(node_id)

        if existing_node is None:
            raise RuntimeError(
                f"ノード '{node_id}' の作成に失敗しました。\n"
                f"プラグイン '{plugin_name}' がインストールされているか確認してください。"
            )

    # MediaIn → Reframe → MediaOut の接続（新規・既存問わず必ず実行）
    try:
        media_in  = comp.FindTool("MediaIn1")
        media_out = comp.FindTool("MediaOut1")
        if media_in and media_out:
            existing_node.Input = media_in.Output
            media_out.Input = existing_node.Output
            logger.info(f"接続: MediaIn1 → {node_id} → MediaOut1")
    except Exception as e:
        logger.warning(f"ノード自動接続に失敗（手動接続が必要な場合あり）: {e}")

    # パラメータ名を動的に確定
    yaw_result   = _find_param(existing_node, yaw_cands)
    pitch_result = _find_param(existing_node, pitch_cands)
    fov_result   = _find_param(existing_node, fov_cands)

    yaw_name   = yaw_result[0]   if yaw_result   else yaw_cands[0]
    pitch_name = pitch_result[0] if pitch_result else pitch_cands[0]
    fov_name   = fov_result[0]   if fov_result   else fov_cands[0]

    logger.info(f"パラメータ確定: Yaw='{yaw_name}' Pitch='{pitch_name}' FOV='{fov_name}'")
    return existing_node, yaw_name, pitch_name, fov_name


def inspect_plugin_parameters(node):
    """デバッグ用: ノードの全パラメータを列挙する"""
    inputs = node.GetInputList() or {}
    logger.debug(f"--- パラメータ一覧 ({len(inputs)} 件) ---")
    for k, v in inputs.items():
        try:
            name = v.Name if hasattr(v, "Name") else str(k)
            logger.debug(f"  [{k}] {name}")
        except Exception:
            logger.debug(f"  [{k}] {v}")


# ─────────────────────────────────────────────────────────────────────────────
# キーフレーム打刻（ステップ変化）
# ─────────────────────────────────────────────────────────────────────────────

def apply_step_keyframes(
    comp,
    ofx_node,
    yaw_param_name: str,
    switching_points: Dict[int, float],
    initial_yaw: float,
) -> int:
    """
    二重キーフレーム打刻法でステップ変化を実現する。

    フレーム N-1: 旧 Yaw 値（アンカー、線形補間を阻止）
    フレーム N  : 新 Yaw 値（瞬時スイッチ）

    Args:
        comp:             Fusion Composition オブジェクト
        ofx_node:         Fusion ノードオブジェクト
        yaw_param_name:   Yaw パラメータ名
        switching_points: { frame_N: new_yaw_value }
        initial_yaw:      初期 Yaw 値

    Returns:
        打刻したキーフレーム総数
    """
    param = getattr(ofx_node, yaw_param_name, None)
    if param is None:
        raise RuntimeError(
            f"パラメータ '{yaw_param_name}' がノードに存在しません。\n"
            f"inspect_plugin_parameters() でパラメータ一覧を確認してください。"
        )

    # アニメーション化（スプライン追加）
    if param.GetConnectedOutput() is None:
        try:
            comp.Lock()
            setattr(ofx_node, yaw_param_name, comp.BezierSpline())
            comp.Unlock()
            param = getattr(ofx_node, yaw_param_name)
            logger.info(f"  '{yaw_param_name}' をアニメーション化しました")
        except Exception as e:
            logger.warning(f"  アニメーション化失敗: {e}")
            comp.Unlock()

    # Fusion Compの開始フレームを取得（大抵は0.0）
    render_start = int(comp.GetAttrs("COMPN_RenderStart") or 0)
    keyframe_count = 0

    # フレーム0に初期値を設定
    try:
        param[render_start] = float(initial_yaw)
        keyframe_count += 1
        logger.info(f"  初期Yaw: frame={render_start}, yaw={initial_yaw:+.1f}°")
    except Exception as e:
        logger.warning(f"  初期Yaw設定失敗: {e}")

    current_yaw = initial_yaw

    for frame_n, new_yaw in sorted(switching_points.items()):
        abs_prev = render_start + frame_n - 1
        abs_curr = render_start + frame_n

        # アンカー（N-1フレーム）
        if abs_prev >= render_start:
            try:
                param[abs_prev] = float(current_yaw)
                keyframe_count += 1
            except Exception as e:
                logger.warning(f"  [Frame {abs_prev}] アンカー打刻失敗: {e}")

        # スイッチポイント（Nフレーム）
        try:
            param[abs_curr] = float(new_yaw)
            keyframe_count += 1
            logger.info(f"  スイッチ: frame={abs_curr}, {current_yaw:+.1f}°→{new_yaw:+.1f}°")
        except Exception as e:
            logger.warning(f"  [Frame {abs_curr}] スイッチ打刻失敗: {e}")

        current_yaw = new_yaw

    logger.info(f"キーフレーム打刻完了: 合計 {keyframe_count} 個")
    return keyframe_count


def set_static_params(ofx_node, pitch_param_name: str, fov_param_name: str,
                      pitch: float = 0.0, fov: float = 90.0):
    """Pitch・FOV を静的な固定値として設定する"""
    for name, value in [(pitch_param_name, pitch), (fov_param_name, fov)]:
        param = getattr(ofx_node, name, None)
        if param is not None:
            try:
                param[0] = float(value)
                logger.info(f"  {name} = {value}")
            except Exception as e:
                logger.warning(f"  {name} 設定失敗: {e}")
