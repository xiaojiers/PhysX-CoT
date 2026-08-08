"""Evaluate kinematic plausibility from pre-composed GT + A-E grid videos."""

import argparse
import base64
import json
import os
from typing import List

import cv2
from openai import OpenAI


SYSTEM_PROMPT = """
You are given a single grid video <GRID_VIDEO> and basic information of the object.
Layout (2 rows x 3 columns), with zero-based indices (row, col):
Top row:    (0,0)=GT, (0,1)=A, (0,2)=B
Bottom row: (1,0)=C,  (1,1)=D, (1,2)=E

Task:
1) Slice the grid accordingly and analyze GT: determine its category and motion pattern.
2) For A/B/C/D/E, evaluate similarity to GT in motion and geometry after mentally aligning
   each candidate via rigid transform.
3) Ignore material, texture, color, lighting, and pure orientation/viewpoint differences.
4) Rank A/B/C/D/E by similarity to GT. Use each candidate's agreement with GT as the basis.
5) Return exactly one JSON object with this schema:
{
  "A": {"geometry_rank": x, "motion_rank": x},
  "B": {"geometry_rank": x, "motion_rank": x},
  "C": {"geometry_rank": x, "motion_rank": x},
  "D": {"geometry_rank": x, "motion_rank": x},
  "E": {"geometry_rank": x, "motion_rank": x}
}
"""


def read_video_frames(path: str, target_width: int = 512, target_height: int = 512) -> List[str]:
    frames = []
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_jump = max(1, int(round(fps / 3.0)))
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_jump == 0:
            resized = cv2.resize(frame, (target_width, target_height))
            _, buffer = cv2.imencode(".jpg", resized)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
    video.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM kinematic ranking on grid videos")
    parser.add_argument("--video_dir", default="./evaluation_video_kine/grids")
    parser.add_argument("--gt_json_dir", default="./dataset/physxnet/finaljson")
    parser.add_argument("--out_dir", default="./evaluation_video_kine/results")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "gemini-2.5-flash-lite"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.zetatechs.com/v1"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or API_KEY before running kinematic VLM evaluation.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    os.makedirs(args.out_dir, exist_ok=True)
    error_dir = os.path.join(args.out_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)

    filenames = sorted(filename for filename in os.listdir(args.video_dir) if filename.endswith(".mp4"))
    if args.max_samples:
        filenames = filenames[: args.max_samples]
    completed = skipped = failed = 0
    for filename in filenames:
        name = os.path.splitext(filename)[0]
        out_path = os.path.join(args.out_dir, name + ".json")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            with open(os.path.join(args.gt_json_dir, name + ".json")) as fp:
                gt_info = json.load(fp)
            frames = read_video_frames(os.path.join(args.video_dir, filename))
            if not frames:
                raise ValueError("no frames read from grid video")
            object_info = (
                f"This is a video of {gt_info.get('object_name', 'an articulated object')}. "
                "Evaluate geometry and kinematic reasonableness against GT."
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": object_info},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + frame},
                            }
                            for frame in frames
                        ],
                    ],
                },
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
            )
            with open(out_path, "w") as fp:
                fp.write(response.choices[0].message.content)
            completed += 1
            print(f"[ok] {name}: {len(frames)} frames")
        except Exception as error:
            failed += 1
            with open(os.path.join(error_dir, name + ".txt"), "w") as fp:
                fp.write(str(error))
            print(f"[fail] {name}: {error}")
    print(f"completed={completed}, existing={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
