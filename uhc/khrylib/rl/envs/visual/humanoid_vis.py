import numpy as np
import mujoco_py

from uhc.khrylib.rl.envs.common import mujoco_env


class HumanoidVisEnv(mujoco_env.MujocoEnv):
    def __init__(self, vis_model_file, nframes=6, focus=True):
        mujoco_env.MujocoEnv.__init__(self, vis_model_file, nframes)

        self.set_cam_first = set()
        self.focus = focus

    def step(self, a):
        return np.zeros((10, 1)), 0, False, dict()

    def reset_model(self):
        c = 0
        self.set_state(
            self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return None

    def sim_forward(self):
        self.sim.forward()

    def set_video_path(
        self, image_path="/tmp/image_%07d.png", video_path="/tmp/video_%07d.mp4"
    ):
        self.viewer._image_path = image_path
        self.viewer._video_path = video_path

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        if self.focus:
            self.viewer.cam.lookat[:2] = self.data.qpos[:2]
            self.viewer.cam.lookat[2] = 0.8
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 30
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.elevation = -10
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def reload_sim_model(self, xml_str):
        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers
        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}
        self._get_viewer("human")._hide_overlay = True
        self.reset()
        print("Reloading Vis Sim")
