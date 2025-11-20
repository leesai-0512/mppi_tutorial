# mujoco_manager.py
import mujoco

class MujocoManager:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data
    
    def get_joint_names(self):
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        return names
