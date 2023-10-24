from uhc.khrylib.rl.envs.visual.humanoid_vis import HumanoidVisEnv
import glfw
import math


class Visualizer:
    def __init__(self, vis_file):
        self.fr = 0
        self.num_fr = 0
        self.T_arr = [1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]
        self.T = 12
        self.paused = False
        self.reverse = False
        self.repeat = False
        self.vis_file = vis_file

        self.env_vis = HumanoidVisEnv(vis_file, 1, focus=False)

        self.env_vis._get_viewer("human")._hide_overlay = True
        self.env_vis.set_custom_key_callback(self.key_callback)

    def data_generator(self):
        raise NotImplementedError

    def update_pose(self):
        raise NotImplementedError

    def key_callback(self, key, action, mods):

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_F:
            self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_Q:
            self.data = next(self.data_gen, None)
            if self.data is None:
                print("end of data!!")
                exit()
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_W:
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_E:
            self.fr = self.num_fr - 1
            self.update_pose()
        elif key == glfw.KEY_G:
            self.repeat = not self.repeat
            self.update_pose()

        elif key == glfw.KEY_S:
            self.reverse = not self.reverse
        elif key == glfw.KEY_RIGHT:
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_pose()
        elif key == glfw.KEY_LEFT:
            if self.fr > 0:
                self.fr -= 1
            self.update_pose()
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        else:
            return False
        return True

    def render(self):
        self.env_vis.render()

    def show_animation(self):
        self.t = 0
        while True:
            if self.t >= math.floor(self.T):
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                    elif self.repeat:
                        self.fr = 0
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                self.update_pose()
                self.t = 0
            self.render()
            if not self.paused:
                self.t += 1
