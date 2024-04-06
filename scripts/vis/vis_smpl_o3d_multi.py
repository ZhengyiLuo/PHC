import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import matplotlib as mpl

paused, reset, recording, image_list, writer, control, curr_zoom = False, False, False, [], None, None, 0.01


def pause_func(action):
    global paused
    paused = not paused
    print(f"Paused: {paused}")
    return True


def reset_func(action):
    global reset
    reset = not reset
    print(f"Reset: {reset}")
    return True


def record_func(action):
    global recording, writer
    if not recording:
        fps = 30
        curr_video_file_name = "test.mp4"
        writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
    elif not writer is None:
        writer.close()
        writer = None

    recording = not recording

    print(f"Recording: {recording}")
    return True


def capture_func(action):
    global capture

    capture = not capture

    return True


def zoom_func(action):
    global control, curr_zoom
    curr_zoom = curr_zoom * 0.9
    control.set_zoom(curr_zoom)
    print(f"Reset: {reset}")
    return True


mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

Name = "getting_started"
Title = "Getting Started"

data_dir = "data/smpl"
smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

# pkl_dir = "output/renderings/smpl_ego_long_8-2023-01-20-11:28:00.pkl"
# pkl_dir = "output/renderings/smpl_im_comp_8-2023-02-05-15:36:14.pkl"
pkl_dir = "output/renderings/smpl_im_comp_pnn_3-2023-03-07-14:31:50.pkl"
Name = pkl_dir.split("/")[-1].split(".")[0]
pkl_data = joblib.load(pkl_dir)
mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]

# data_file = "data/quest/home1_isaac.pkl"
# sk_tree = SkeletonTree.from_mjcf(f"/tmp/smpl/test_good.xml")
# motion_lib = MotionLibSMPLTest("data/quest/home1_isaac.pkl", [7, 3, 22, 17],torch.device("cpu"))
# motion_lib.load_motions(skeleton_trees=[sk_tree],
#                         gender_betas=[torch.zeros(17)] ,
#                         limb_weights=[np.zeros(10)] ,
#                         random_sample=False)


def main():
    global reset, paused, recording, image_list, control, capture
    capture = False
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    # opt.background_color = [0.5, 0.5, 0.5]  # gray color

    smpl_meshes = dict()
    items = list(pkl_data.items())
    color_picker = [mpl.colormaps['YlOrBr'], mpl.colormaps['Reds']]
    idx = 0
    print(len(items))
    for entry_key, data_seq in tqdm(items):

        gender, beta = data_seq['betas'][0], data_seq['betas'][1:]
        if gender == 0:
            smpl_parser = smpl_parser_n
        elif gender == 1:
            smpl_parser = smpl_parser_m
        else:
            smpl_parser = smpl_parser_f

        pose_quat, trans = data_seq['body_quat'].numpy()[::2], data_seq['trans'].numpy()[::2]
        skeleton_tree = SkeletonTree.from_dict(data_seq['skeleton_tree'])
        offset = skeleton_tree.local_translation[0]
        root_trans_offset = trans - offset.numpy()

        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)

        global_rot = sk_state.global_rotation
        B, J, N = global_rot.shape
        pose_quat = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
        B_down = pose_quat.shape[0]
        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=False)
        local_rot = new_sk_state.local_rotation
        pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(B_down, -1, 3)
        pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(B_down, -1)
        with torch.no_grad():
            vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_trans=torch.from_numpy(root_trans_offset), th_betas=torch.from_numpy(beta[None,]))

        vertices = vertices.numpy()
        faces = smpl_parser.faces
        max_frames = vertices.shape[0]

        for i in tqdm(range(0, max_frames - 270, 15)):
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            # vertex_colors = np.array([35, 102, 218]) / 256 * (1 - i / vertices.shape[0])
            # vertex_colors = color_picker[idx] * ((i + 60) / max_frames)
            vertex_colors = color_picker[idx](1 - ((i / max_frames) * 0.5 + 0.2))[:3]

            smpl_mesh.paint_uniform_color(vertex_colors)
            smpl_mesh.compute_vertex_normals()
            vis.add_geometry(smpl_mesh)

        for i in tqdm(range(max_frames - 280, max_frames - 230, 15)):
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            # vertex_colors = np.array([35, 102, 218]) / 256 * (1 - i / vertices.shape[0])
            # vertex_colors = color_picker[idx] * ((i + 60) / max_frames)
            vertex_colors = color_picker[idx](1 - ((i / max_frames) * 0.5 + 0.2))[:3]

            smpl_mesh.paint_uniform_color(vertex_colors)
            smpl_mesh.compute_vertex_normals()
            vis.add_geometry(smpl_mesh)

        for i in tqdm(range(max_frames - 230, max_frames - 200, 8)):
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            # vertex_colors = np.array([35, 102, 218]) / 256 * (1 - i / vertices.shape[0])
            # vertex_colors = color_picker[idx] * ((i + 60) / max_frames)
            vertex_colors = color_picker[idx](1 - ((i / max_frames) * 0.5 + 0.2))[:3]

            smpl_mesh.paint_uniform_color(vertex_colors)
            smpl_mesh.compute_vertex_normals()
            vis.add_geometry(smpl_mesh)

        for i in tqdm(range(max_frames - 200, max_frames - 30, 15)):
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            # vertex_colors = np.array([35, 102, 218]) / 256 * (1 - i / vertices.shape[0])
            # vertex_colors = color_picker[idx] * ((i + 60) / max_frames)
            vertex_colors = color_picker[idx](1 - ((i / max_frames) * 0.5 + 0.2))[:3]

            smpl_mesh.paint_uniform_color(vertex_colors)
            smpl_mesh.compute_vertex_normals()
            vis.add_geometry(smpl_mesh)

        for i in tqdm(range(max_frames - 30, max_frames, 15)):
            smpl_mesh = o3d.geometry.TriangleMesh()
            smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
            smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            # vertex_colors = np.array([35, 102, 218]) / 256 * (1 - i / vertices.shape[0])
            vertex_colors = color_picker[idx](1 - ((i / max_frames) * 0.5 + 0.2))[:3]

            smpl_mesh.paint_uniform_color(vertex_colors)
            smpl_mesh.compute_vertex_normals()
            vis.add_geometry(smpl_mesh)

        smpl_meshes[entry_key] = {
            'mesh': smpl_mesh,
            "vertices": vertices,
        }
        idx += 1

    box = o3d.geometry.TriangleMesh()
    ground_size, height = 50, 0.01
    box = box.create_box(width=ground_size, height=height, depth=ground_size)
    box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
    box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.vertex_colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]]).repeat(8, axis=0))
    # box.paint_uniform_color(vertex_colors)
    vis.add_geometry(box)

    control = vis.get_view_control()

    control.unset_constant_z_far()
    control.unset_constant_z_near()
    i = 0

    vis.register_key_callback(32, pause_func)
    vis.register_key_callback(82, reset_func)
    vis.register_key_callback(76, record_func)
    vis.register_key_callback(67, capture_func)
    vis.register_key_callback(90, zoom_func)

    control.set_up(np.array([0, 0, 1]))
    control.set_front(np.array([-5, 0, 1]))
    control.set_lookat(np.array([0, 0, 1]))

    control.set_zoom(0.1)
    dt = 1 / 30

    tracker_pos = pkl_data['0_0']['ref_body_pos_subset'][::2].cpu().numpy()

    while True:
        vis.poll_events()

        # if not paused:
        #     i = (i + 1)

        if reset:
            i = 0
            reset = False
        if recording:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)
            writer.append_data(rgb)
        if capture:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)
            name = input("Enter image name:")
            img_name = f"output/renderings/iccv2023/{name}.png"
            print("Captruing image to {}".format(img_name))
            cv2.imwrite(img_name, np.asarray(rgb)[..., ::-1])
            capture = False

        vis.update_renderer()


if __name__ == "__main__":
    main()