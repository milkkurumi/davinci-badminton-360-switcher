"""
badminton_switcher.py - DaVinci Resolve バドミントン専用360°自動スイッチャー
メインエントリポイント

【実行方法】
  DaVinci Resolve の上部メニュー:
    Workspace > Scripts > Edit > badminton_switcher

【重要】
  DaVinci Resolve のスクリプト実行環境では以下がグローバルとして自動注入される:
    resolve / fusion / comp / project / timeline
  このスクリプトはこれらを直接利用する。bmd.scriptapp() は使用しない。

【動作モード】
  config.json の execution_options.phase1_dummy_mode:
    true  → Phase 1: ダミー座標でキーフレーム打刻テスト（TrackNet不要）
    false → Phase 3: TrackNetV3 による本番解析 + キーフレーム打刻
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# プロジェクトルートの解決
# DaVinci Resolve の Scripts フォルダにコピーして実行した場合でも、
# 元のプロジェクトフォルダを正しく参照できるよう固定パスを設定する。
# ─────────────────────────────────────────────────────────────────────────────

# ▼▼▼ このパスをあなたの環境に合わせて変更してください ▼▼▼
_PROJECT_ROOT = Path(r"c:\Users\iwami\Desktop\dev\DaVinci Resolve バドミントン専用360°動画")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# フォールバック: スクリプトがプロジェクト直下にある場合は __file__ を使う
try:
    _SCRIPT_DIR = Path(__file__).parent.resolve()
    if (_SCRIPT_DIR / "config.json").exists():
        _PROJECT_ROOT = _SCRIPT_DIR
except NameError:
    # DaVinci Resolve の環境では __file__ が定義されないことがあるため無視する
    pass

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# venv の site-packages を sys.path に追加（DaVinci 環境でも torch 等を使えるように）
_VENV_SITE = _PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

# TrackNetV3 ディレクトリも追加
_TRACKNETV3_DIR = _PROJECT_ROOT / "TrackNetV3"
if _TRACKNETV3_DIR.exists() and str(_TRACKNETV3_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKNETV3_DIR))


def main():
    # ─── DaVinci グローバル変数を取得 ──────────────────────────────────
    # DaVinci Resolve の Scripts メニューから実行すると
    # resolve / fusion / comp / project / timeline が自動注入される
    _g          = globals()
    _resolve    = _g.get("resolve")
    _fusion     = _g.get("fusion")
    _comp       = _g.get("comp")       # 現在開いている Fusion Composition
    _project    = _g.get("project")
    _timeline   = _g.get("timeline")

    from src.utils import setup_logger, load_config, ensure_output_dir
    from src.switching_logic import SwitchingStateMachine, generate_dummy_coords
    from src.davinci_api import (
        get_resolve_globals,
        get_target_timeline_item,
        get_clip_info,
        get_timeline_fps,
        get_fusion_comp,
        find_or_create_reframe_node,
        apply_step_keyframes,
        set_static_params,
        inspect_plugin_parameters,
    )

    # ─── 設定ロード ──────────────────────────────────────────────────────
    config = load_config(_PROJECT_ROOT / "config.json")

    cam  = config["camera_setup"]
    algo = config["algorithm_tuning"]
    opt  = config["execution_options"]
    proj = config.get("projection", {})

    debug_mode    = opt.get("debug_mode", False)
    phase1_dummy  = opt.get("phase1_dummy_mode", True)
    apply_step_kf = opt.get("apply_step_keyframes", True)

    logger = setup_logger(debug=debug_mode)
    ensure_output_dir(_PROJECT_ROOT)

    logger.info("=" * 60)
    logger.info("  バドミントン専用360°自動スイッチャー 起動")
    logger.info(f"  モード: {'Phase 1 (ダミー)' if phase1_dummy else 'Phase 3 (本番)'}")
    logger.info("=" * 60)

    # ─── Step 1: DaVinci Resolve グローバル確認 ──────────────────────
    logger.info("[Step 1/5] DaVinci Resolve グローバル変数を確認中...")

    # resolve が取得できているか確認
    if _resolve is None and _fusion is not None:
        try:
            _resolve = _fusion.GetResolve()
        except Exception:
            pass

    if _resolve is None:
        raise RuntimeError(
            "resolve グローバル変数が取得できません。\n"
            "Edit ページの Workspace > Scripts > Edit から実行してください。"
        )
    logger.info(f"  resolve: OK ({_resolve})")

    # ─── Step 2: タイムライン・FPS 取得 ──────────────────────────────
    logger.info("[Step 2/5] タイムライン情報を取得中...")

    if _timeline is None:
        try:
            pm = _resolve.GetProjectManager()
            _project  = pm.GetCurrentProject()
            _timeline = _project.GetCurrentTimeline()
        except Exception as e:
            raise RuntimeError(f"タイムラインの取得に失敗しました: {e}")

    fps = get_timeline_fps(_timeline)
    logger.info(f"  FPS: {fps}")

    # ─── Step 3: 対象クリップ取得 ────────────────────────────────────
    logger.info("[Step 3/5] 対象クリップを取得中...")
    timeline_item = get_target_timeline_item(_timeline)
    clip_info     = get_clip_info(timeline_item)
    clip_start    = clip_info["start_frame"]
    clip_duration = clip_info["duration"]

    # ─── Step 4: シャトル座標リスト生成 ─────────────────────────────
    logger.info("[Step 4/5] シャトル座標を取得中...")

    if phase1_dummy:
        # ── Phase 1: ダミーモード ─────────────────────────────────
        logger.info("  ダミーモード: config.json の phase1_dummy_switches を使用")
        dummy_switches = opt.get("phase1_dummy_switches", [100, 250, 400, 600])
        coord_list = generate_dummy_coords(
            total_frames=clip_duration,
            switch_frames=dummy_switches,
        )
    else:
        # ── Phase 3: TrackNetV3 本番解析 ──────────────────────────────
        from src.shuttle_tracker import ShuttleTracker, save_coords_to_csv
        from src.utils import normalize_x as _norm_x
        import cv2

        tkn = config.get("tracknet", {})
        tracknet_path = str(_PROJECT_ROOT / tkn.get("model_path", "models/TrackNet_best.pt"))
        inpaint_path  = str(_PROJECT_ROOT / tkn.get("inpaintnet_path", "models/InpaintNet_best.pt"))
        use_inpaint   = tkn.get("use_inpaintnet", True)

        # タイムラインアイテムからメディアファイルパスを取得
        video_path = None
        try:
            mpi = timeline_item.GetMediaPoolItem()
            if mpi:
                video_path = mpi.GetClipProperty("File Path")
        except Exception:
            pass

        if not video_path:
            raise RuntimeError(
                "解析対象の動画パスを取得できませんでした。\n"
                "メディアプールとタイムラインが正しくリンクされているか確認してください。"
            )

        logger.info(f"  解析対象: {video_path}")

        tracker = ShuttleTracker(
            tracknet_path=tracknet_path,
            inpaintnet_path=inpaint_path if use_inpaint else None,
            use_inpaintnet=use_inpaint,
            court_yaw=cam.get("court_center_yaw", 0.0),
            court_pitch=cam.get("court_center_pitch", -10.0),
            court_fov=proj.get("gnomonic_fov_deg", 100.0),
            gnomonic_out_size=(
                proj.get("gnomonic_out_width", 1280),
                proj.get("gnomonic_out_height", 720),
            ),
        )

        raw_results = tracker.analyze_video(
            video_path,
            tmp_flat_path=str(_PROJECT_ROOT / "output" / "tmp_flat.mp4"),
        )

        csv_path = str(_PROJECT_ROOT / opt.get("output_csv_path", "output/shuttle_coords.csv"))
        save_coords_to_csv(raw_results, csv_path, fps=fps)

        cap_tmp = cv2.VideoCapture(video_path)
        orig_w  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_tmp.release()

        coord_list = []
        for frame_idx, coord, conf in raw_results:
            if coord is not None:
                x_norm = _norm_x(coord[0], orig_w)
                coord_list.append((frame_idx, x_norm, conf))
            else:
                coord_list.append((frame_idx, None, 0.0))

    # ─── Step 4.5: スイッチング判定 ──────────────────────────────────
    sm = SwitchingStateMachine(
        yaw_angle_a=cam["yaw_angle_a"],
        yaw_angle_b=cam["yaw_angle_b"],
        deadzone_min=algo["deadzone_min"],
        deadzone_max=algo["deadzone_max"],
        min_hold_frames=algo["min_hold_frames"],
        confidence_threshold=algo["shuttle_confidence_threshold"],
        lost_tolerance_frames=algo.get("lost_tolerance_frames", 10),
    )

    for frame_idx, x_norm, conf in coord_list:
        sm.process(frame_idx, x_norm, conf)

    switching_points = sm.get_switching_points()
    initial_yaw      = cam["yaw_angle_a"]

    logger.info(f"  切り替え回数: {sm.stats.total_switches} 回")
    logger.info(f"  {sm.stats.summary()}")

    # ─── Step 5: DaVinci Resolve キーフレーム打刻 ────────────────────
    logger.info("[Step 5/5] Fusion にキーフレームを打刻中...")

    # Fusion Comp 取得（グローバルの comp を優先使用）
    comp = get_fusion_comp(timeline_item, current_comp=_comp)

    # プラグイン選択
    plugin_name = cam.get("fx_plugin_name", "DJIReframe")
    fallback    = cam.get("fx_plugin_fallback", "PanoMap")

    try:
        ofx_node, yaw_pname, pitch_pname, fov_pname = find_or_create_reframe_node(
            comp, plugin_name
        )
        logger.info(f"  プラグイン: {plugin_name} (Yaw={yaw_pname})")
    except RuntimeError as e:
        logger.warning(f"  {plugin_name} 取得失敗 → {fallback} を使用: {e}")
        ofx_node, yaw_pname, pitch_pname, fov_pname = find_or_create_reframe_node(
            comp, fallback
        )
        logger.info(f"  プラグイン: {fallback} (Yaw={yaw_pname})")

    if debug_mode:
        inspect_plugin_parameters(ofx_node)

    set_static_params(
        ofx_node, pitch_pname, fov_pname,
        pitch=cam.get("default_pitch", 0.0),
        fov=cam.get("default_fov", 90.0),
    )

    if apply_step_kf:
        kf_count = apply_step_keyframes(
            comp=comp,
            ofx_node=ofx_node,
            yaw_param_name=yaw_pname,
            switching_points=switching_points,
            initial_yaw=initial_yaw,
        )
        logger.info(f"  打刻完了: {kf_count} キーフレーム")
    else:
        logger.info("  apply_step_keyframes=false のためスキップ")

    # ─── 完了 ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  ✅ 完了！")
    logger.info(f"  切り替え回数: {sm.stats.total_switches} 回")
    logger.info("  Fusion ページでタイムラインをスクラブして確認してください。")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
