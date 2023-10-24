import os
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
import math
import numpy as np
from uhc.utils.transformation import quaternion_from_matrix
from scipy.spatial.transform import Rotation as sRot

TEMPLATE_FILE = "phc/data/assets/mjcf/humanoid_template_local.xml"

GAINS = {
    "L_Hip": [500, 50, 1, 500],
    "L_Knee": [500, 50, 1, 500],
    "L_Ankle": [400, 40, 1, 500],
    "L_Toe": [200, 20, 1, 500],
    "R_Hip": [500, 50, 1, 500],
    "R_Knee": [500, 50, 1, 500],
    "R_Ankle": [400, 40, 1, 500],
    "R_Toe": [200, 20, 1, 500],
    "Torso": [1000, 100, 1, 500],
    "Spine": [1000, 100, 1, 500],
    "Chest": [1000, 100, 1, 500],
    "Neck": [100, 10, 1, 250],
    "Head": [100, 10, 1, 250],
    "L_Thorax": [400, 40, 1, 500],
    "L_Shoulder": [400, 40, 1, 500],
    "L_Elbow": [300, 30, 1, 150],
    "L_Wrist": [100, 10, 1, 150],
    "L_Hand": [100, 10, 1, 150],
    "R_Thorax": [400, 40, 1, 150],
    "R_Shoulder": [400, 40, 1, 250],
    "R_Elbow": [300, 30, 1, 150],
    "R_Wrist": [100, 10, 1, 150],
    "R_Hand": [100, 10, 1, 150],
}


class Bone:

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []
        self.offset = np.zeros(3)
        self.sites = []

        # inferred info
        self.pos = np.zeros(3)
        self.bone = np.zeros(3)
        self.ends = []


class Skeleton:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.root = None
        self.equalities = None
        self.exclude_contacts = None
        self.collision_groups = None
        self.simple_geom = False
        self.buffer_dict = {"njmax": "2500", "nconmax": "500"}

    def forward_bones(self, bone):
        if bone.parent:
            # bone.pos = bone.parent.pos + bone.offset
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bones(bone_c)

    def load_from_offsets(
        self,
        offsets,
        parents,
        axes,
        channels,
        jrange,
        sites,
        scale,
        equalities,
        hull_dict,
        exclude_contacts=None,
        collision_groups=None,
        conaffinity=None,
        simple_geom=False,
        color_dict=None,
        real_weight=False,
        replace_feet=True,
    ):
        if exclude_contacts is None:
            exclude_contacts = []
        if collision_groups is None:
            collision_groups = {}
        self.exclude_contacts = exclude_contacts
        self.collision_groups = {}
        self.conaffinity = {}
        self.color_dict = color_dict  # giving color to the meshes
        self.real_weight = real_weight
        self.real_weight_porpotion_capsules = True
        self.replace_feet = replace_feet
        self.hull_dict = hull_dict

        for group, bones in collision_groups.items():
            for bone in bones:
                self.collision_groups[bone] = group

        for group, bones in conaffinity.items():
            for bone in bones:
                self.conaffinity[bone] = group

        self.simple_geom = simple_geom

        joint_names = list(offsets.keys())
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.equalities = equalities
        self.len_scale = scale
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.orient = axes[joint_names[0]]
        self.root.pos = offsets[joint_names[0]]
        self.root.sites = sites.get(joint_names[0], [])
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)

        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = channels[joint]
            bone.dof_index = [dof_ind[x[0]] for x in bone.channels]
            bone.offset = offsets[joint] * self.len_scale
            bone.orient = axes[joint]
            bone.lb = np.rad2deg(jrange[joint][:, 0])
            bone.ub = np.rad2deg(jrange[joint][:, 1])
            bone.sites = sites.get(joint, [])
            self.bones.append(bone)
            self.name2bone[joint] = bone

        for bone in self.bones[1:]:
            parent_name = parents[bone.name]
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bones(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.ends.append(bone.pos.copy())
                bone.end = bone.pos.copy() + 0.002
                for c_bone, p_bone in parents.items():
                    if p_bone == bone.name:
                        bone.end += np.array(offsets[c_bone]) * self.len_scale
                        break
            else:
                bone.end = sum([bone_c.pos
                                for bone_c in bone.child]) / len(bone.child)
                for bone_c in bone.child:
                    bone.ends.append(bone_c.pos.copy())

    def write_str(
            self,
            template_fname=TEMPLATE_FILE,
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        tree = self.construct_tree(ref_angles=ref_angles,
                                   offset=offset,
                                   template_fname=template_fname)
        if bump_buffer:
            SubElement(tree.getroot(), "size", self.buffer_dict)
        return etree.tostring(tree, pretty_print=True)

    def write_xml(
            self,
            fname,
            template_fname=TEMPLATE_FILE,
            offset=np.array([0, 0, 0]),
            ref_angles=None,
            bump_buffer=False,
    ):
        tree = self.construct_tree(ref_angles=ref_angles,
                                   offset=offset,
                                   template_fname=template_fname)
        if bump_buffer:
            SubElement(tree.getroot(), "size", self.buffer_dict)
        # create sensors
        # sensor = tree.getroot().find("sensor")
        # for bone in self.bones:
        #     SubElement(sensor, 'framelinvel', {'objtype': 'body', 'objname': bone.name})
        # for bone in self.bones:
        #     SubElement(sensor, 'frameangvel', {'objtype': 'body', 'objname': bone.name})
        # for bone in self.bones:
        #     SubElement(sensor, 'framelinvel', {'objtype': 'xbody', 'objname': bone.name})

        tree.write(fname, pretty_print=True)

    def construct_tree(
            self,
            template_fname=TEMPLATE_FILE,
            offset=np.array([0, 0, 0]),
            ref_angles=None,
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")

        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create meshes
        asset = tree.getroot().find("asset")
        for bone in self.bones:
            if os.path.exists(f"{self.model_dir}/geom/{bone.name}.stl"):
                attr = {
                    "file":
                    f"{self.model_dir.split('/')[-1]}/geom/{bone.name}.stl",
                    "name": f"{bone.name}_mesh"
                }
                # geom_relative_path = f'../mesh/smpl/{self.model_dir.split("/")[-1]}'
                # attr = {"file": f"{geom_relative_path}/geom/{bone.name}.stl", "name": f"{bone.name}_mesh"}
                SubElement(asset, "mesh", attr)

        # create actuators
        actuators = tree.getroot().find("actuator")

        joints = worldbody.findall(".//joint")
        for joint in joints:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "1"
            SubElement(actuators, "motor", attr)

        # create exclude contacts
        c_node = tree.getroot().find("contact")
        for bname1, bname2 in self.exclude_contacts:
            attr = {"body1": bname1, "body2": bname2}
            SubElement(c_node, "exclude", attr)
        # create equalities
        eq_node = tree.getroot().find("equality")
        for eq_joints in self.equalities.values():
            for j1 in range(len(eq_joints) - 1):
                for j2 in range(j1 + 1, len(eq_joints)):
                    jname1, jcoeff1 = eq_joints[j1]
                    jname2, jcoeff2 = eq_joints[j2]
                    coeff = jcoeff1 / jcoeff2
                    attr = {
                        "joint1": jname1,
                        "joint2": jname2,
                        "polycoef": f"0 {coeff:.6f} 0 0 0",
                    }
                    SubElement(eq_node, "joint", attr)
        return tree

    def write_xml_bodynode(self, bone, parent_node, offset, ref_angles):
        if self.real_weight:
            base_density = 1000
        else:
            base_density = 500

        attr = dict()
        attr["name"] = bone.name
        attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
        quat = quaternion_from_matrix(bone.orient)
        attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(*quat)
        node = SubElement(parent_node, "body", attr)

        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            # j_attr["limited"] = "false"
            # j_attr["type"] = "free"
            # j_attr["armature"] = "0.02"
            # j_attr["damping"] = "50"
            # j_attr["stiffness"] = "500"
            # j_attr["frictionloss"] = "0"

            SubElement(node, "freejoint", j_attr)
        else:

            for i in range(len(bone.channels)):
                ind = bone.dof_index[i]
                axis = bone.orient[:, ind]
                j_attr = dict()

                j_attr["name"] = bone.name + "_" + bone.channels[i]
                j_attr["type"] = "hinge"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos +
                                                                   offset))
                j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)

                j_attr["stiffness"] = str(GAINS[bone.name][0])
                j_attr["damping"] = str(GAINS[bone.name][1])
                if bone.name in ["L_Ankle", "R_Ankle"]:
                    j_attr["armature"] = "0.01"
                else:
                    j_attr["armature"] = "0.02"

                if i < len(bone.lb):
                    j_attr["range"] = "{0:.4f} {1:.4f}".format(
                        bone.lb[i], bone.ub[i])
                else:
                    j_attr["range"] = "-180.0 180.0"
                if j_attr["name"] in ref_angles.keys():
                    j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"
                SubElement(node, "joint", j_attr)

        # write sites
        for s_name, s_pos, s_quat in bone.sites:
            s_attr = {"name": s_name}
            s_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(s_pos + offset))
            s_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(*s_quat)
            s_attr["type"] = "sphere"
            s_attr["size"] = "0.03"
            SubElement(node, "site", s_attr)

        geom_path = f"{self.model_dir}/geom/{bone.name}.stl"
        if os.path.exists(geom_path):
            g_attr = {"type": "mesh", "mesh": f"{bone.name}_mesh"}
            if bone.name in self.collision_groups.keys():
                g_attr["density"] = str(base_density)

                g_attr["contype"] = str(self.collision_groups[bone.name])
                g_attr["conaffinity"] = str(self.conaffinity[bone.name])

                # g_attr["solimp"] = "0.9 0.95 0.001 0.5 2"
                # g_attr["solref"] = "0.02 1"
                # g_attr["size"] = str(10)
                # g_attr["friction"] = "0.000000000005 0.000000000005 0.1"
                if not self.color_dict is None:
                    g_attr["rgba"] = self.color_dict[bone.name]

                if bone.name in ["L_Ankle", "R_Ankle", "L_Toe", "R_Toe"
                                 ] and self.replace_feet:
                    g_attr = {}
                    hull_params = self.hull_dict[bone.name]
                    min_verts, max_verts = hull_params['norm_verts'].min(
                        axis=0), hull_params['norm_verts'].max(axis=0)
                    size = max_verts - min_verts

                    bone_end = bone.end
                    pos = (max_verts + min_verts) / 2
                    size /= 2

                    if bone.name == "L_Toe" or bone.name == "R_Toe":
                        parent_min, parent_max = self.hull_dict[
                            bone.parent.name]['norm_verts'].min(
                                axis=0), self.hull_dict[
                                    bone.parent.name]['norm_verts'].max(axis=0)
                        parent_pos = (parent_max + parent_min) / 2

                        pos[2] = parent_min[2] - bone.pos[2] + size[
                            2]  # To get toe to be at the same height as the parent
                        pos[1] = parent_pos[1] - bone.pos[
                            1]  # To get toe to be at the y as the parent

                    rot = np.array([1, 0, 0, 0])
                    if self.real_weight_porpotion_capsules:
                        g_attr["density"] = str(
                            (hull_params['volume'] /
                             (size[0] * size[1] * size[2] * 8)) * base_density)
                    g_attr["type"] = "box"
                    g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)
                    g_attr["size"] = "{0:.4f} {1:.4f} {2:.4f}".format(*size)
                    g_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(
                        *rot)

            SubElement(node, "geom", g_attr)
        else:
            for end in bone.ends:
                g_attr = dict()
                e1 = bone.pos + offset
                e2 = end + offset
                v = e2 - e1
                if np.linalg.norm(v) > 1e-6:
                    v /= np.linalg.norm(v)
                    e1 += v * 0.02
                    e2 -= v * 0.02
                    g_attr["type"] = "capsule"
                    g_attr[
                        "fromto"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(
                            *np.concatenate([e1, e2]))
                else:
                    g_attr["type"] = "sphere"
                    g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*bone.pos)
                g_attr["size"] = "0.0300" if self.simple_geom else "0.0100"
                if not self.simple_geom:
                    g_attr["contype"] = "0"
                    g_attr["conaffinity"] = "0"
                elif bone.name in self.collision_groups.keys():
                    group = str(self.collision_groups[bone.name])
                    g_attr["contype"] = group
                    g_attr["conaffinity"] = group
                SubElement(node, "geom", g_attr)

        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node, offset, ref_angles)

            