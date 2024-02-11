import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
from uhc.khrylib.utils.transformation import euler_matrix
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
import math
import re
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags

TEMPLATE_FILE = "phc/data/assets/mjcf/humanoid_template_local.xml"
GEOM_TYPES = {
    'Pelvis': 'sphere',
    'L_Hip': 'capsule',
    'L_Knee': 'capsule',
    'L_Ankle': 'box',
    'L_Toe': 'box',
    'R_Hip': 'capsule',
    'R_Knee': 'capsule',
    'R_Ankle': 'box',
    'R_Toe': 'box',
    'Torso': 'capsule',
    'Spine': 'capsule',
    'Chest': 'capsule',
    'Neck': 'capsule',
    'Head': 'sphere',
    'L_Thorax': 'capsule',
    'L_Shoulder': 'capsule',
    'L_Elbow': 'capsule',
    'L_Wrist': 'capsule',
    'L_Hand': 'sphere',
    # 'L_Hand': 'box',
    'R_Thorax': 'capsule',
    'R_Shoulder': 'capsule',
    'R_Elbow': 'capsule',
    'R_Wrist': 'capsule',
    'R_Hand': 'sphere',
    # 'R_Hand': 'box',
}

GAINS = {
    "L_Hip": [800, 80, 1, 500],
    "L_Knee": [800, 80, 1, 500],
    "L_Ankle": [800, 80, 1, 500],
    "L_Toe": [500, 50, 1, 500],
    "R_Hip": [800, 80, 1, 500],
    "R_Knee": [800, 80, 1, 500],
    "R_Ankle": [800, 80, 1, 500],
    "R_Toe": [500, 50, 1, 500],
    "Torso": [1000, 100, 1, 500],
    "Spine": [1000, 100, 1, 500],
    "Chest": [1000, 100, 1, 500],
    "Neck": [500, 50, 1, 250],
    "Head": [500, 50, 1, 250],
    "L_Thorax": [500, 50, 1, 500],
    "L_Shoulder": [500, 50, 1, 500],
    "L_Elbow": [500, 50, 1, 150],
    "L_Wrist": [300, 30, 1, 150],
    "L_Hand": [300, 30, 1, 150],
    "R_Thorax": [500, 50, 1, 150],
    "R_Shoulder": [500, 50, 1, 250],
    "R_Elbow": [500, 50, 1, 150],
    "R_Wrist": [300, 30, 1, 150],
    "R_Hand": [300, 30, 1, 150],
}


class Bone:

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)


class Skeleton:

    def __init__(self):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None

    def forward_bvh(self, bone):
        bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)

    def load_from_offsets(
        self,
        offsets,
        parents,
        scale,
        jrange,
        hull_dict,
        exclude_bones=None,
        channels=None,
        spec_channels=None,
        upright_start=False,
        remove_toe=False,
        freeze_hand= False,
        real_weight_porpotion_capsules=False,
        real_weight_porpotion_boxes = False, 
        real_weight=False,
        big_ankle=False,
        box_body = False, 
    ):
        if channels is None:
            channels = ["x", "y", "z"]
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        self.hull_dict = hull_dict
        self.upright_start = upright_start
        self.remove_toe = remove_toe
        self.real_weight_porpotion_capsules = real_weight_porpotion_capsules
        self.real_weight_porpotion_boxes = real_weight_porpotion_boxes
        self.real_weight = real_weight
        self.big_ankle = big_ankle
        self.freeze_hand = freeze_hand
        self.box_body = box_body
        joint_names = list(filter(lambda x: all([t not in x for t in exclude_bones]), offsets.keys()))
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = scale
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = channels
        self.name2bone[self.root.name] = self.root
        self.root.offset = offsets[self.root.name]
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint

            bone.channels = (spec_channels[joint] if joint in spec_channels.keys() else channels)
            bone.dof_index = [dof_ind[x] for x in bone.channels]
            bone.offset = np.array(offsets[joint]) * self.len_scale
            bone.lb = np.rad2deg(jrange[joint][:, 0])
            bone.ub = np.rad2deg(jrange[joint][:, 1])

            self.bones.append(bone)
            self.name2bone[joint] = bone
        for bone in self.bones[1:]:
            parent_name = parents[bone.name]
            # print(parent_name)
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p
        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = bone.pos.copy() + 0.002
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def write_xml(
            self,
            fname,
            template_fname=TEMPLATE_FILE,
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.size_buffer = {}
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")

        for joint in joints[1:]:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "500"
            SubElement(actuators, "motor", attr)
        if bump_buffer:
            SubElement(tree.getroot(), "size", {"njmax": "700", "nconmax": "700"})
        tree.write(fname, pretty_print=True)

    def write_str(
            self,
            template_fname=TEMPLATE_FILE,
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.size_buffer = {}
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")
        for joint in joints:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "500"
            SubElement(actuators, "motor", attr)
        if bump_buffer:
            SubElement(tree.getroot(), "size", {"njmax": "700", "nconmax": "700"})

        return etree.tostring(tree, pretty_print=False)

    def write_xml_bodynode(self, bone, parent_node, offset, ref_angles):
        attr = dict()
        attr["name"] = bone.name
        attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
        node = SubElement(parent_node, "body", attr)

        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            SubElement(node, "freejoint", j_attr)
        else:
            for i in range(len(bone.dof_index)):
                ind = bone.dof_index[i]
                axis = bone.orient[:, ind]
                j_attr = dict()
                j_attr["name"] = bone.name + "_" + self.dof_name[ind]
                j_attr["type"] = "hinge"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
                j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)
                j_attr["stiffness"] = str(GAINS[bone.name][0])
                j_attr["damping"] = str(GAINS[bone.name][1])
                j_attr["armature"] = "0.02"

                if i < len(bone.lb):
                    j_attr["range"] = "{0:.4f} {1:.4f}".format(bone.lb[i], bone.ub[i])
                else:
                    j_attr["range"] = "-180.0 180.0"
                if j_attr["name"] in ref_angles.keys():
                    j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"

                SubElement(node, "joint", j_attr)

        # write geometry
        g_attr = dict()
        if not self.freeze_hand:
            GEOM_TYPES['L_Hand'] = 'box'
            GEOM_TYPES['R_Hand'] = 'box'

        if self.box_body:
            GEOM_TYPES['Head'] = 'box'
            GEOM_TYPES['Pelvis'] = 'box'
        
        g_attr["type"] = GEOM_TYPES[bone.name]
        g_attr["contype"] = "1"
        g_attr["conaffinity"] = "1"
        if self.real_weight:
            base_density = 1000
        else: 
            base_density = 500
        g_attr["density"] = str(base_density)
        e1 = np.zeros(3)
        e2 = bone.end.copy() + offset
        if bone.name in ["Torso", "Chest", "Spine"]:
            seperation = 0.45
        else:
            seperation = 0.2

        # if bone.name in ["L_Hip"]:
        #     seperation = 0.3

        e1 += e2 * seperation
        e2 -= e2 * seperation
        hull_params = self.hull_dict[bone.name]

        if g_attr["type"] == "capsule":
            g_attr["fromto"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(*np.concatenate([e1, e2]))

            side_len = np.linalg.norm(e2 - e1)
            # radius = 0.067
            # V = np.pi * radius ** 2 * ((4/3) * radius + side_len)

            roots = np.polynomial.polynomial.Polynomial([-hull_params['volume'], 0, side_len * np.pi, 4 / 3 * np.pi]).roots()
            real_valued = roots.real[abs(roots.imag) < 1e-5]
            real_valued = real_valued[real_valued > 0]
            if bone.name in ["Torso", "Spine", "L_Hip", "R_Hip"]:
                real_valued *= 0.7  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:  # If shift is enabled, shift the weight based on teh shrinkage factor
                    g_attr["density"] = str((1 / 0.7**2) * base_density)

            if bone.name in ["Chest"]:
                real_valued *= 0.7  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.7**2) * base_density)

            if bone.name in ["L_Knee", 'R_Knee']:
                real_valued *= 0.9  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.9**2) * base_density)

            # if bone.name in ["Spine"]:
            # real_valued *= 0.01 # ZL Hack: shrinkage

            # g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
            g_attr["size"] = "{0:.4f}".format(*real_valued)

        elif g_attr["type"] == "box":
            pos = (e1 + e2) / 2
            min_verts = hull_params['norm_verts'].min(axis=0).values
            size = (hull_params['norm_verts'].max(axis=0).values - min_verts).numpy()
            if self.upright_start:
                if bone.name == "L_Toe" or bone.name == "R_Toe":
                    size[0] = hull_params['volume'] / (size[2] * size[0])
                else:
                    size[2] = hull_params['volume'] / (size[1] * size[0])
            else:
                size[1] = hull_params['volume'] / (size[2] * size[0])
            size /= 2
            
            if bone.name == "L_Toe" or bone.name == "R_Toe":
                if self.upright_start:
                    pos[2] = -bone.pos[2] / 2 - self.size_buffer[bone.parent.name][2] + size[2]  # To get toe to be at the same height as the parent
                    pos[1] = -bone.pos[1] / 2  # To get toe to be at the same x as the parent
                else:
                    pos[1] = -bone.pos[1] / 2 - self.size_buffer[bone.parent.name][1] + size[1]  # To get toe to be at the same height as the parent
                    pos[0] = -bone.pos[0] / 2  # To get toe to be at the same x as the parent

                if self.remove_toe:
                    size /= 20  # Smaller toes...
                    pos[1] = 0
                    pos[0] = 0
            bone_dir = bone.end / np.linalg.norm(bone.end)
            if not self.remove_toe:
                rot = np.array([1, 0, 0, 0])
            else:
                rot = sRot.from_euler("xyz", [0, 0, np.arctan(bone_dir[1] / bone_dir[0])]).as_quat()[[3, 0, 1, 2]]

            if self.big_ankle:
                # Big ankle override
                g_attr = {}
                hull_params = self.hull_dict[bone.name]
                min_verts, max_verts = hull_params['norm_verts'].min(axis=0).values, hull_params['norm_verts'].max(axis=0).values
                size = max_verts - min_verts

                bone_end = bone.end
                pos = (max_verts + min_verts) / 2
                size /= 2
                
                if bone.name == "L_Toe" or bone.name == "R_Toe":
                    parent_min, parent_max = self.hull_dict[bone.parent.name]['norm_verts'].min(axis=0).values, self.hull_dict[bone.parent.name]['norm_verts'].max(axis=0).values
                    parent_pos = (parent_max + parent_min) / 2
                    if self.upright_start:
                        pos[2] = parent_min[2] - bone.pos[2] + size[2]  # To get toe to be at the same height as the parent
                        pos[1] = parent_pos[1] - bone.pos[1]  # To get toe to be at the y as the parent
                    else:
                        pos[1] = parent_min[1] - bone.pos[1] + size[1]  # To get toe to be at the same height as the parent
                        pos[0] = parent_pos[0] - bone.pos[0]  # To get toe to be at the y as the parent

                rot = np.array([1, 0, 0, 0])
                
                g_attr["type"] = "box"
                g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)
                g_attr["size"] = "{0:.4f} {1:.4f} {2:.4f}".format(*size)
                g_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(*rot)

            if bone.name == "Pelvis":
                size /= 1.75  # ZL Hack: shrinkage
                
            if bone.name == "Head":
                size[0] /= 1.5  # ZL Hack: shrinkage
                size[1] /= 1.5  # ZL Hack: shrinkage
                
            if self.real_weight_porpotion_boxes:
                g_attr["density"] = str((hull_params['volume'] / (size[0] * size[1] * size[2] * 8).item()) * base_density)

            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)
            g_attr["size"] = "{0:.4f} {1:.4f} {2:.4f}".format(*size)
            g_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(*rot)
            self.size_buffer[bone.name] = size

        elif g_attr["type"] == "sphere":
            pos = np.zeros(3)
            radius = np.cbrt(hull_params['volume'] * 3 / (4 * np.pi))
            if bone.name in ["Pelvis"]:
                radius *= 0.6  # ZL Hack: shrinkage
                if self.real_weight_porpotion_capsules:
                    g_attr["density"] = str((1 / 0.6**3) * base_density)

            g_attr["size"] = "{0:.4f}".format(radius)
            # g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)

        SubElement(node, "geom", g_attr)

        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node, offset, ref_angles)
