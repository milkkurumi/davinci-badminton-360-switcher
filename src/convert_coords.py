import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.projection import gnomonic_coord_to_erp_x_normalized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="TrackNetV3 output CSV (e.g. flat_test_ball.csv)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=-10.0)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=1280, help="Flat video width")
    parser.add_argument("--height", type=int, default=720, help="Flat video height")
    parser.add_argument("--erp_width", type=int, default=5760, help="Original 360 video width")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    switches = []
    
    # 連続したフレームをサンプリング（例: 30フレーム(0.6秒)ごとにカメラの向きを変える）
    # 細かく振りすぎるとカメラがブレるのでスムージングが必要
    
    last_added_frame = -999
    
    for i, row in df.iterrows():
        vis = row['Visibility']
        if vis == 1:
            frame = int(row['Frame'])
            
            # 30フレーム(約0.6秒)ごとのみキーフレームを打つ
            if frame - last_added_frame >= 30:
                cx = float(row['X'])
                cy = float(row['Y'])
                
                # 360度空間での正規化X座標 (0.0〜1.0) を取得
                erp_x_norm = gnomonic_coord_to_erp_x_normalized(
                    cx=cx, cy=cy,
                    out_width=args.width, out_height=args.height,
                    yaw_deg=args.yaw, pitch_deg=args.pitch, fov_deg=args.fov,
                    erp_width=args.erp_width
                )
                
                # 0.0〜1.0 の座標系を -0.5〜0.5 (画面中央を0) に変換
                center_x = erp_x_norm - 0.5
                
                switches.append({
                    "frame": frame,
                    "target_cam": "Center", # 仮
                    "center_x": center_x
                })
                
                last_added_frame = frame
                
    # JSONに書き出し
    out_dict = {
        "phase2_auto_switches": switches
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4, ensure_ascii=False)
        
    print(f"生成されたスイッチ数: {len(switches)}")
    print(f"保存完了: {args.output}")

if __name__ == "__main__":
    main()
