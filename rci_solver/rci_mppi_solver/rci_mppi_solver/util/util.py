import numpy as np
import pinocchio as pin
import jax
import jax.numpy as jnp



class Util:
    def __init__(self):
        pass

    @staticmethod
    def vectorToSE3(vec):
        assert vec.shape == (12,)
        translation = vec[:3]
        rotation_flat = vec[3:]
        rotation = rotation_flat.reshape((3, 3))
        return pin.SE3(rotation, translation)
    

    @staticmethod
    def SE3Tovector(T):

        vec = np.zeros(12)
        vec[:3] = T.translation
        vec[3:] = T.rotation.flatten()
        return vec
    
    @staticmethod
    def errorInSE3(T, Tdes):

        T_err = T.inverse() * Tdes
        error = pin.Motion.Zero()
        error.linear = T_err.translation
        error.angular = pin.log3(T_err.rotation)
        return error.vector
    
    @staticmethod
    def rotmat_to_quat(R):
        m00 = R[..., 0, 0]
        m01 = R[..., 0, 1]
        m02 = R[..., 0, 2]
        m10 = R[..., 1, 0]
        m11 = R[..., 1, 1]
        m12 = R[..., 1, 2]
        m20 = R[..., 2, 0]
        m21 = R[..., 2, 1]
        m22 = R[..., 2, 2]

        trace = m00 + m11 + m22
        qw = 0.5 * jnp.sqrt(jnp.clip(trace + 1.0, a_min=1e-6))
        qx = 0.5 * jnp.sign(m21 - m12) * jnp.sqrt(jnp.clip(1.0 + m00 - m11 - m22, a_min=1e-6))
        qy = 0.5 * jnp.sign(m02 - m20) * jnp.sqrt(jnp.clip(1.0 - m00 + m11 - m22, a_min=1e-6))
        qz = 0.5 * jnp.sign(m10 - m01) * jnp.sqrt(jnp.clip(1.0 - m00 - m11 + m22, a_min=1e-6))

        q = jnp.stack([qx, qy, qz, qw], axis=-1)  # (..., 4)
        return q
        
    @staticmethod
    def goal_checker(T_goal, T_current, pos_tol=2.0e-2, ori_tol=5e-1):
        T_goal_np = np.array(jax.device_get(T_goal))

        # 위치 오차
        pos_goal = T_goal_np[:3, 3]
        pos_current = T_current.translation
        pos_error = pos_goal - pos_current
        pos_error_norm = np.linalg.norm(pos_error)
        # 자세 오차
        R_goal = T_goal_np[:3, :3]
        R_current = T_current.rotation
        R_error = R_goal.T @ R_current
        rot_vec = pin.log3(R_error)
        ori_error_angle = np.linalg.norm(rot_vec)

        is_goal_reached = (pos_error_norm < pos_tol) and (ori_error_angle < ori_tol)
        # print("pos_error_norm: ",pos_error_norm)
        # print("ori_error_angle: ",ori_error_angle)
        return is_goal_reached
    

    @staticmethod
    def goal_checker_only_pos(T_goal, T_current, pos_tol=2.0e-2):
        T_goal_np = np.array(jax.device_get(T_goal))

        # 위치 오차
        pos_goal = T_goal_np[:3, 3]
        pos_current = T_current.translation
        pos_error = pos_goal - pos_current
        pos_error_norm = np.linalg.norm(pos_error)

        is_goal_reached = (pos_error_norm < pos_tol) 
        print("pos_error_norm: ",pos_error_norm)
        # print("ori_error_angle: ",ori_error_angle)
        return is_goal_reached
    

    @staticmethod
    def is_grasp(finger_q, finger_v):
        grasp = finger_q[0] < 0.02 and finger_q[1] < 0.2
        grasp = grasp and finger_v[0] < 0.0001 and finger_v[1] < 0.0001
        return grasp