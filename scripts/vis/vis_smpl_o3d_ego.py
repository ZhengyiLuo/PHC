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
import matplotlib as mpl
from datetime import datetime
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
        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        curr_video_file_name = f"output/renderings/o3d/{curr_date_time}-test.mp4"
        writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
    elif not writer is None:
        writer.close()
        writer = None

    recording = not recording

    print(f"Recording: {recording}")
    return True


def zoom_func(action):
    global control, curr_zoom
    curr_zoom = curr_zoom * 0.9
    control.set_zoom(curr_zoom)
    print(f"Reset: {reset}")
    return True

colorpicker = mpl.colormaps['Blues']
mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

Name = "getting_started"
Title = "Getting Started"

data_dir = "data/smpl"
smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

pkl_dir = "output/renderings/smpl_im_comp_pnn_1_1_demo-2023-03-13-14:28:47.pkl"
Name = pkl_dir.split("/")[-1].split(".")[0]
pkl_data = joblib.load(pkl_dir)
mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]

# data_file = "data/quest/home1_isaac.pkl"
# sk_tree = SkeletonTree.from_mjcf(f"/tmp/smpl/test_good.xml")
# motion_lib.load_motions(skeleton_trees=[sk_tree],
#                         gender_betas=[torch.zeros(17)] ,
#                         limb_weights=[np.zeros(10)] ,
#                         random_sample=False)


def main():
    global reset, paused, recording, image_list, control
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    ############ Loading texture ############
    texture_path = "/hdd/zen/data/SURREAL/smpl_data/"
    faces_uv = np.load(os.path.join(texture_path, 'final_faces_uv_mapping.npy'))
    uv_sampler = torch.from_numpy(faces_uv.reshape(-1, 2, 2, 2))
    uv_sampler = uv_sampler.view(-1, 13776, 2 * 2, 2)
    texture_img_path_male = osp.join(texture_path, "textures", "male")
    texture_img_path_female = osp.join(texture_path, "textures", "female")
    ############ Loading texture ############

    smpl_meshes = dict()
    items = list(pkl_data.items())

    for entry_key, data_seq in tqdm(items):
        gender, beta = data_seq['betas'][0], data_seq['betas'][1:]
        if gender == 0:
            smpl_parser = smpl_parser_n
            texture_image_path = texture_img_path_male
        elif gender == 1:
            smpl_parser = smpl_parser_m
            texture_image_path = texture_img_path_male
        else:
            smpl_parser = smpl_parser_f
            texture_image_path = texture_img_path_female

        pose_quat, trans = data_seq['body_quat'].numpy()[::2], data_seq['trans'].numpy()[::2]
        if pose_quat.shape[0] < 200:
            continue
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

        vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_trans=torch.from_numpy(root_trans_offset), th_betas=torch.from_numpy(beta[None,]))
        vertices = vertices.numpy()
        faces = smpl_parser.faces
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
        smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)

        ######################## Smampling texture ########################
        batch_size = 1
        # uv_sampler = uv_sampler.repeat(batch_size, 1, 1, 1)  ##torch.Size([B, 13776, 4, 2])
        # full_path = "nongrey_male_0237.jpg"
        # # full_path = random.choice(os.listdir(texture_image_path))
        # texture_image = plt.imread(osp.join(texture_image_path, full_path))

        # texture_image = np.transpose(texture_image, (2, 0, 1))
        # texture_image = torch.from_numpy(texture_image).float() / 255.0
        # textures = torch.nn.functional.grid_sample(texture_image[None,], uv_sampler, align_corners=True)  #torch.Size([N, 3, 13776, 4])
        # textures = textures.permute(0, 2, 3, 1)  #torch.Size([N, 13776, 4, 3])
        # # textures = textures.view(-1, 13776, 2, 2, 3) #torch.Size([N, 13776, 2, 2, 3])
        # textures = textures.squeeze().numpy()

        # vertex_colors = {}
        # for idx, f in enumerate(faces):
        #     colors = textures[idx]
        #     for vidx, vid in enumerate(f):
        #         vertex_colors[vid] = colors[vidx]
        # vertex_colors = np.array([vertex_colors[i] for i in range(len(vertex_colors))])
        # smpl_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        vertex_colors = colorpicker(0.6)[:3]
        smpl_mesh.paint_uniform_color(vertex_colors)
        
        smpl_mesh.compute_vertex_normals()
        ######################## Smampling texture ########################
        vis.add_geometry(smpl_mesh)
        smpl_meshes[entry_key] = {
            'mesh': smpl_mesh,
            "vertices": vertices,
        }
        break

    box = o3d.geometry.TriangleMesh()
    ground_size, height = 50, 0.01
    box = box.create_box(width=ground_size, height=height, depth=ground_size)
    box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
    box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
    box.compute_vertex_normals()
    box.vertex_colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]]).repeat(8, axis=0))

    spheres = []
    for _ in range(24):
        sphere = o3d.geometry.TriangleMesh()
        sphere = sphere.create_sphere(radius=0.05)
        sphere.compute_vertex_normals()
        sphere.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.1, 0.9, 0.1]]).repeat(len(sphere.vertices), axis=0))
        spheres.append(sphere)

    sphere_pos = np.zeros([24, 3])
    [vis.add_geometry(sphere) for sphere in spheres]
    vis.add_geometry(box)

    control = vis.get_view_control()

    control.unset_constant_z_far()
    control.unset_constant_z_near()
    i = 0
    N = vertices.shape[0]

    vis.register_key_callback(32, pause_func)
    vis.register_key_callback(82, reset_func)
    vis.register_key_callback(76, record_func)
    vis.register_key_callback(90, zoom_func)

    control.set_up(np.array([0, 0, 1]))
    control.set_front(np.array([1, 0, 0]))
    control.set_lookat(vertices[0, 0])

    control.set_zoom(0.5)
    dt = 1 / 30

    to_isaac_mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    tracker_pos = pkl_data['0_0']['ref_body_pos_subset'][::2].cpu().numpy()
    tracker_pos = np.matmul(tracker_pos, to_isaac_mat.T)

    while True:
        vis.poll_events()
        for smpl_mesh_key, smpl_mesh_data in smpl_meshes.items():
            verts = smpl_mesh_data["vertices"]
            smpl_mesh_data["mesh"].vertices = o3d.utility.Vector3dVector(verts[i % verts.shape[0]])
            vis.update_geometry(smpl_mesh_data["mesh"])

            # motion_res = motion_lib.get_motion_state(torch.tensor([0]), torch.tensor([(i % verts.shape[0]) * dt]))
            # curr_pos = motion_res['rg_pos'][0, [13, 18 ,23]].numpy()
            curr_pos = tracker_pos[i % verts.shape[0]]

            for idx, s in enumerate(spheres):
                s.translate((curr_pos - sphere_pos)[idx])
                vis.update_geometry(s)
            sphere_pos = curr_pos
            # sphere.translate(verts[0, 0])
            # vis.update_geometry(sphere)

        if not paused:
            i = (i + 1)

        if reset:
            i = 0
            reset = False
        if recording:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)
            writer.append_data(rgb)

        vis.update_renderer()


if __name__ == "__main__":
    main()