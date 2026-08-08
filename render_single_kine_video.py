#!/usr/bin/env python3
"""Render a single URDF kinematic animation with PyBullet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio
import numpy as np
import pybullet as p
import pybullet_data


MOVABLE_TYPES = {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}


def place_above_ground(body_id: int, margin: float = 0.3) -> None:
    min_z = float("inf")
    for link_idx in [-1] + list(range(p.getNumJoints(body_id))):
        aabb_min, _ = p.getAABB(body_id, link_idx)
        if aabb_min is not None:
            min_z = min(min_z, aabb_min[2])
    if min_z != float("inf"):
        position, orientation = p.getBasePositionAndOrientation(body_id)
        p.resetBasePositionAndOrientation(
            body_id,
            (position[0], position[1], position[2] + margin - min_z),
            orientation,
        )


def body_aabb(body_id: int) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for link_idx in [-1] + list(range(p.getNumJoints(body_id))):
        aabb_min, aabb_max = p.getAABB(body_id, link_idx)
        if aabb_min is None or aabb_max is None:
            continue
        mins.append(aabb_min)
        maxs.append(aabb_max)
    if not mins:
        return np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0])
    return np.min(np.asarray(mins, dtype=float), axis=0), np.max(np.asarray(maxs, dtype=float), axis=0)


def joint_summary(body_id: int) -> list[dict[str, object]]:
    items = []
    for index in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, index)
        joint_type = int(info[2])
        items.append(
            {
                "index": index,
                "name": info[1].decode("utf-8", errors="replace"),
                "type": joint_type,
                "type_name": {
                    p.JOINT_REVOLUTE: "revolute",
                    p.JOINT_PRISMATIC: "prismatic",
                    p.JOINT_SPHERICAL: "spherical",
                    p.JOINT_PLANAR: "planar",
                    p.JOINT_FIXED: "fixed",
                }.get(joint_type, str(joint_type)),
                "lower": float(info[8]),
                "upper": float(info[9]),
            }
        )
    return items


def target_for_joint(info, step: int, sim_hz: int, offset: int) -> float:
    lower, upper = float(info[8]), float(info[9])
    phase = 2 * math.pi * (step / sim_hz) * 0.5 + offset * 0.35
    if lower < upper and abs(lower) < 100 and abs(upper) < 100:
        center = 0.5 * (lower + upper)
        amplitude = 0.45 * (upper - lower)
        return center + amplitude * math.sin(phase)
    return 0.8 * math.sin(phase)


def render(
    urdf_path: Path,
    out_mp4: Path,
    yaw: float,
    pitch: float,
    ground_margin: float,
    fps: int,
    duration: float,
    width: int,
    height: int,
    overwrite: bool,
) -> dict[str, object]:
    if out_mp4.exists() and not overwrite:
        raise FileExistsError(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    sim_hz = 240
    cid = p.connect(p.DIRECT)
    writer = None
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / sim_hz)
        p.loadURDF("plane.urdf")
        robot = p.loadURDF(str(urdf_path), useFixedBase=True)
        place_above_ground(robot, margin=ground_margin)

        aabb_min, aabb_max = body_aabb(robot)
        center = 0.5 * (aabb_min + aabb_max)
        extent = aabb_max - aabb_min
        radius = max(float(np.linalg.norm(extent)), 0.5)
        distance = max(2.0, min(8.0, radius * 1.7))
        target_z = float(center[2])
        target = [float(center[0]), float(center[1]), target_z]

        joints = joint_summary(robot)
        movable = [item["index"] for item in joints if item["type"] in MOVABLE_TYPES]
        view = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)
        projection = p.computeProjectionMatrixFOV(60, width / float(height), 0.01, 20)

        writer = imageio.get_writer(out_mp4, fps=fps, quality=9)
        for step in range(int(duration * sim_hz)):
            for offset, joint_index in enumerate(movable):
                info = p.getJointInfo(robot, joint_index)
                target_position = target_for_joint(info, step, sim_hz, offset)
                p.setJointMotorControl2(
                    robot,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_position,
                    force=500,
                )
            p.stepSimulation()
            if step % max(1, sim_hz // fps) == 0:
                _, _, rgba, _, _ = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER)
                writer.append_data(np.asarray(rgba, dtype=np.uint8)[..., :3])

        return {
            "urdf": str(urdf_path),
            "video": str(out_mp4),
            "yaw": yaw,
            "pitch": pitch,
            "ground_margin": ground_margin,
            "fps": fps,
            "duration": duration,
            "frame_size": [width, height],
            "camera_target": target,
            "camera_distance": distance,
            "aabb_min": aabb_min.tolist(),
            "aabb_max": aabb_max.tolist(),
            "joint_count": len(joints),
            "movable_joint_count": len(movable),
            "joints": joints,
        }
    finally:
        if writer is not None:
            writer.close()
        p.disconnect(cid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out_mp4", type=Path, required=True)
    parser.add_argument("--yaw", type=float, default=-45.0)
    parser.add_argument("--pitch", type=float, default=-25.0)
    parser.add_argument("--ground_margin", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = render(
        urdf_path=args.urdf.resolve(),
        out_mp4=args.out_mp4.resolve(),
        yaw=args.yaw,
        pitch=args.pitch,
        ground_margin=args.ground_margin,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
        overwrite=args.overwrite,
    )
    manifest = args.out_mp4.with_suffix(".json")
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"video": result["video"], "movable_joint_count": result["movable_joint_count"]}, indent=2))


if __name__ == "__main__":
    main()
