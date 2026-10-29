"""
debug_time.py - Fusion ページの時間軸・キーフレーム調査
Workspace > Scripts > Edit > debug_time から実行
"""
import sys

_comp = globals().get("comp")
if _comp is None:
    print("[NG] comp が取得できません。")
    sys.exit(1)

print("=" * 50)
print("  Fusion コンポジション 時間軸調査")
print("=" * 50)

# コンポジションの時間属性を取得
attrs = _comp.GetAttrs()
render_start = attrs.get("COMPN_RenderStart", "?")
render_end   = attrs.get("COMPN_RenderEnd", "?")
global_start = attrs.get("COMPN_GlobalStart", "?")
global_end   = attrs.get("COMPN_GlobalEnd", "?")

print(f"  Render Range : {render_start} 〜 {render_end}")
print(f"  Global Range : {global_start} 〜 {global_end}")

# Transform ノードを探す
print("\n--- Transform ノードの状態確認 ---")
transform = _comp.FindTool("Transform1")
if transform:
    print("  Transform1 発見")
    angle_param = transform.Angle
    if angle_param:
        # アニメーションがついているか（スプラインがあるか）
        is_animated = angle_param.GetConnectedOutput() is not None
        print(f"  Angle はアニメーション化されているか？: {is_animated}")
        
        # 現在のキーフレームを取得
        keys = angle_param.GetKeyFrames()
        print(f"  現在のキーフレーム数: {len(keys) if keys else 0}")
        if keys:
            for k, v in list(keys.items())[:10]:
                print(f"    Frame {k} = {v}")
            if len(keys) > 10:
                print("    ... (省略)")
        else:
            print("  [NG] キーフレームが1つも存在しません！")
            
            # 手動でアニメーション修飾子（BezierSpline）を追加してみるテスト
            print("\n  → 修飾子を追加して打刻を再試行します")
            try:
                # AddModifier を使用せずに Bezier スプラインを強制
                _comp.Lock()
                transform.Angle = _comp.BezierSpline()
                transform.Angle[render_start] = 0.0
                transform.Angle[render_start + 100] = 45.0
                transform.Angle[render_start + 200] = -45.0
                _comp.Unlock()
                
                new_keys = transform.Angle.GetKeyFrames()
                print(f"  再試行後のキーフレーム数: {len(new_keys) if new_keys else 0}")
                for k, v in list(new_keys.items()):
                    print(f"    Frame {k} = {v}")
            except Exception as e:
                print(f"  再試行エラー: {e}")
                _comp.Unlock()

else:
    print("  [NG] Transform1 ノードが見つかりません。")

print("\n==================================================")
