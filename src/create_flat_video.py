import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.projection import extract_court_region

def main():
    parser = argparse.ArgumentParser(description="Convert 360 video to flat court video")
    parser.add_argument("--video", required=True, help="Input 360 video")
    parser.add_argument("--output", required=True, help="Output flat video")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=-10.0)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start_frame", type=int, default=0, help="開始フレーム")
    parser.add_argument("--num_frames", type=int, default=-1, help="処理フレーム数")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    process_frames = total_frames - args.start_frame
    if args.num_frames > 0:
        process_frames = min(process_frames, args.num_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (args.width, args.height))

    print(f"Processing {process_frames} frames (from frame {args.start_frame})...")
    
    for i in tqdm(range(process_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        flat_frame = extract_court_region(
            frame,
            court_yaw=args.yaw,
            court_pitch=args.pitch,
            fov_deg=args.fov,
            out_size=(args.width, args.height)
        )
        out.write(flat_frame)

    cap.release()
    out.release()
    print(f"Saved flat video to {args.output}")

if __name__ == "__main__":
    main()
