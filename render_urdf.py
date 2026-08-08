import os, math, time
import numpy as np
import imageio
import pybullet as p
import pybullet_data
def place_above_ground(body_id: int, margin: float = 0.01):

    n_j = p.getNumJoints(body_id)
    min_z = float('inf')
    for link_idx in [-1] + list(range(n_j)):
        aabb_min, aabb_max = p.getAABB(body_id, link_idx)
        if aabb_min is None:  
            continue
        min_z = min(min_z, aabb_min[2])

    if min_z == float('inf'):
        return  

    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    dz = (margin - min_z)

    new_pos = (base_pos[0], base_pos[1], base_pos[2] + dz)
    p.resetBasePositionAndOrientation(body_id, new_pos, base_orn)

evalpath='./test_demo'
namelist=os.listdir(evalpath)
savepath='./evaluation_video_physxanything'
os.makedirs(os.path.join(savepath), exist_ok=True)

for name in namelist:
    try:
        URDF_PATH = os.path.join(evalpath,name,'basic.urdf') 

        OUT_MP4   = os.path.join(savepath,name+'.mp4')
        if os.path.exists(OUT_MP4):
                continue
        FPS       = 30
        SIM_HZ    = 240            
        DURATION  = 2.0           
        W, H      = 512, 512

        
        cid = p.connect(p.DIRECT)  
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_HZ)

        plane = p.loadURDF("plane.urdf")
        robot = p.loadURDF(URDF_PATH, useFixedBase=True)
        place_above_ground(robot, margin=0.3) 

        n_j = p.getNumJoints(robot)
        joint_idxs = [j for j in range(n_j)
                    if p.getJointInfo(robot, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]

        cam_target = [0, 0, 0.8]
        cam_dist   = 2.0
        cam_yaw    = -45
        cam_pitch  = -25
        view = p.computeViewMatrixFromYawPitchRoll(cam_target, cam_dist, cam_yaw, cam_pitch, 0, 2)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=W/float(H), nearVal=0.01, farVal=10)

        writer = imageio.get_writer(OUT_MP4, fps=FPS, quality=9)  

        steps = int(DURATION * SIM_HZ)
        render_every = int(SIM_HZ // FPS)  
        for t in range(steps):
            for i, j in enumerate(joint_idxs):
                target = 0.8 * math.sin(2 * math.pi * (t / SIM_HZ) * 0.5 + i*0.3)
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=target, force=500)

            p.stepSimulation()

            if t % render_every == 0:
                _, _, rgba, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
                frame = np.uint8(rgba)[..., :3]  
                writer.append_data(frame)

        writer.close()
        p.disconnect()
        print(f"save: {OUT_MP4}")
    except:
        None

