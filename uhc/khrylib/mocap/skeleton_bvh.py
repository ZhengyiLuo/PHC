from uhc.khrylib.utils.transformation import euler_matrix
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
import math
import re
from bvh import Bvh
import numpy as np


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

    def load_from_asf(self, fname, swap_axes=False):
        with open(fname) as f:
            content = f.readlines()

        dof_ind = {"rx": 0, "ry": 1, "rz": 2}
        phase = 0
        lastbone = None
        setting_limit = False

        for line in content:
            line_words = line.split()
            cmd = line_words[0]
            if cmd == ":root":
                phase = 0
                self.root = Bone()
                self.root.id = 0
                self.root.name = "root"
                self.name2bone["root"] = self.root
                self.bones.append(self.root)
                continue
            if cmd == ":bonedata":
                phase = 1
                continue
            if cmd == ":hierarchy":
                phase = 2
                continue

            if phase == 0:
                if cmd == "mass":
                    self.mass_scale = float(line_words[1])
                elif cmd == "length":
                    self.len_scale = 1 / float(line_words[1]) * 0.0254
            if phase == 1:
                if cmd == "begin":
                    lastbone = Bone()
                    self.bones.append(lastbone)
                elif cmd == "end":
                    setting_limit = False
                    lastbone = None
                elif cmd == "id":
                    lastbone.id = len(self.bones) - 1
                elif cmd == "name":
                    lastbone.name = line_words[1]
                    self.name2bone[lastbone.name] = lastbone
                elif cmd == "direction":
                    for i in range(3):
                        lastbone.dir[i] = float(line_words[i + 1])
                    if swap_axes:
                        lastbone.dir[1], lastbone.dir[2] = (
                            -lastbone.dir[2],
                            lastbone.dir[1],
                        )
                elif cmd == "length":
                    lastbone.len = float(line_words[1]) * self.len_scale
                elif cmd == "axis":
                    args = [math.radians(float(word)) for word in line_words[1:4]]
                    lastbone.orient = euler_matrix(*args, "sxyz")[:3, :3]
                    if swap_axes:
                        orient = lastbone.orient.copy()
                        lastbone.orient[1, :], lastbone.orient[2, :] = (
                            -orient[2, :],
                            orient[1, :],
                        )
                elif cmd == "dof":
                    for word in reversed(line_words[1:]):
                        if word in dof_ind:
                            ind = dof_ind[word]
                            lastbone.dof_index.append(ind)
                elif cmd == "limits" or setting_limit:
                    lastbone.lb.append(
                        float(re.sub("[(]", " ", line_words[1 - setting_limit]))
                    )
                    lastbone.ub.append(
                        float(re.sub("[)]", " ", line_words[2 - setting_limit]))
                    )
                    setting_limit = True

            if phase == 2:
                if cmd != "begin" and cmd != "end":
                    bone_p = self.name2bone[line_words[0]]
                    for child_name in line_words[1:]:
                        bone_c = self.name2bone[child_name]
                        bone_p.child.append(bone_c)
                        bone_c.parent = bone_p

        self.forward_asf(self.root)

    def forward_asf(self, bone):
        if bone.parent:
            bone.pos = bone.parent.end
        bone.end = bone.pos + bone.dir * bone.len
        for bone_c in bone.child:
            self.forward_asf(bone_c)

    def load_from_bvh(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        with open(fname) as f:
            mocap = Bvh(f.read())

        joint_names = list(
            filter(
                lambda x: all([t not in x for t in exclude_bones]),
                mocap.get_joints_names(),
            )
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = 0.0254
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = mocap.joint_channels(self.root.name)
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = (
                spec_channels[joint]
                if joint in spec_channels.keys()
                else mocap.joint_channels(joint)
            )
            bone.dof_index = [dof_ind[x[0].lower()] for x in bone.channels]
            bone.offset = np.array(mocap.joint_offset(joint)) * self.len_scale
            bone.lb = [-180.0] * 3
            bone.ub = [180.0] * 3
            self.bones.append(bone)
            self.name2bone[joint] = bone

        for bone in self.bones[1:]:
            parent_name = mocap.joint_parent(bone.name).name
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = (
                    bone.pos
                    + np.array(
                        [
                            float(x)
                            for x in mocap.get_joint(bone.name).children[-1]["OFFSET"]
                        ]
                    )
                    * self.len_scale
                )
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def forward_bvh(self, bone):
        if bone.parent:
            bone.pos = bone.parent.pos + bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)

    def load_from_offsets(
        self,
        offsets,
        parents,
        scale,
        exclude_bones=None,
        channels=None,
        spec_channels=None,
    ):
        if channels is None:
            channels = ["x", "y", "z"]
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()

        joint_names = list(
            filter(lambda x: all([t not in x for t in exclude_bones]), offsets.keys())
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = scale
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = channels
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = (
                spec_channels[joint] if joint in spec_channels.keys() else channels
            )
            bone.dof_index = [dof_ind[x] for x in bone.channels]
            bone.offset = np.array(offsets[joint]) * self.len_scale
            if "Elbow" in joint:
                bone.lb = [-720.0] * 3
                bone.ub = [720.0] * 3
            else:
                bone.lb = [-180.0] * 3
                bone.ub = [180.0] * 3

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
        # import pdb
        # pdb.set_trace()
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = bone.pos.copy()
                for c_bone, p_bone in parents.items():
                    if p_bone == bone.name:
                        bone.end += np.array(offsets[c_bone]) * self.len_scale
                        break
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def write_xml(
        self, fname, template_fname, offset=np.array([0, 0, 0]), ref_angles=None
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")
        for joint in joints[1:]:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "1"
            SubElement(actuators, "motor", attr)

        tree.write(fname, pretty_print=True)

    def write_xml_bodynode(self, bone, parent_node, offset, ref_angles):
        attr = dict()
        attr["name"] = bone.name
        attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
        attr["user"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.end + offset))
        node = SubElement(parent_node, "body", attr)

        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
            j_attr["limited"] = "false"
            j_attr["type"] = "free"
            j_attr["armature"] = "0"
            j_attr["damping"] = "0"
            j_attr["stiffness"] = "0"
            SubElement(node, "joint", j_attr)
        else:
            for i in range(len(bone.dof_index)):
                ind = bone.dof_index[i]
                axis = bone.orient[:, ind]
                j_attr = dict()
                j_attr["name"] = bone.name + "_" + self.dof_name[ind]
                j_attr["type"] = "hinge"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
                j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)

                if i < len(bone.lb):
                    j_attr["range"] = "{0:.4f} {1:.4f}".format(bone.lb[i], bone.ub[i])
                else:
                    j_attr["range"] = "-180.0 180.0"
                if j_attr["name"] in ref_angles.keys():
                    j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"

                SubElement(node, "joint", j_attr)

        # write geometry
        if bone.parent is None:
            g_attr = dict()
            g_attr["size"] = "0.0300"
            g_attr["type"] = "sphere"
            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
            SubElement(node, "geom", g_attr)
        else:
            e1 = bone.pos.copy() + offset
            e2 = bone.end.copy() + offset
            v = e2 - e1
            if np.linalg.norm(v) > 1e-6:
                v /= np.linalg.norm(v)
            else:
                v = np.array([0.0, 0.0, 0.2])
            e1 += v * 0.02
            e2 -= v * 0.02
            g_attr = dict()
            g_attr["size"] = "0.0300"
            g_attr["type"] = "capsule"
            g_attr["contype"] = "0"
            g_attr["conaffinity"] = "1"
            g_attr["fromto"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(
                *np.concatenate([e1, e2])
            )
            SubElement(node, "geom", g_attr)

        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node, offset, ref_angles)
