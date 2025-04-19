# =============================== #
#  environment.py
# =============================== #
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matlab.engine
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

class CellFreeMiMoCSIEnv(gym.Env):
    """Gymnasium 环境 —— Cell‑Free Massive‑MIMO 天线激活优化 (PPO)。"""

    metadata = {"render_modes": []}

    def __init__(self,
                 N_AP: int,
                 N_UE: int,
                 max_steps: int = 256,
                 se_threshold: float = 7.0,
                 matlab_proj_path: str | None = None,
                 matlab_startup_msg: bool = True):
        super().__init__()

        # -------------------- 基本参数 -------------------- #
        self.N_AP        = int(N_AP)
        self.N_UE        = int(N_UE)
        self.max_steps   = int(max_steps)
        self.se_thr      = float(se_threshold)
        self.current_step: int = 0

        # -------------------- 日志初始化 -------------------- #
        self.logger = logging.getLogger("CellFreeMiMoCSIEnvLogger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join("./env_logs", now)
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "env_step_log.txt"))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)

        # -------------------- MATLAB 引擎 -------------------- #
        self.eng = matlab.engine.start_matlab()
        proj_path = matlab_proj_path or r'C:\Users\asus\Desktop\master\003-master_thesis\002 - 004_dnn'
        self.eng.addpath(self.eng.genpath(proj_path), nargout=0)
        if matlab_startup_msg:
            print("[MATLAB PATH]", self.eng.path())

        # -------------------- Gym 空间 -------------------- #
        # 动作：每个 AP 选 {0,4,8} 根天线  -> MultiDiscrete([3]*N_AP)
        self.action_space = spaces.MultiDiscrete([3] * self.N_AP)

        # 观测：CSI( N_AP×N_UE ) + 动作( N_AP ) + 4 个标量特征
        state_size = self.N_AP * self.N_UE + self.N_AP + 4
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype=np.float32)

        # -------------------- 上一步缓存 -------------------- #
        self.prev_action = np.zeros(self.N_AP, dtype=np.int32)   # 直接存根数 0/4/8
        self.prev_SE     = 0.0
        self.prev_Ptot   = 0.0

    # ------------------------------------------------------ #
    # reset()                                                #
    # ------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        # 若用户没给 seed，就随机一个并立即生效 (MATLAB 内部根据 seed 放置 UE/AP)
        self.seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**16))
        self.current_step = 0

        # 初始动作：全部 8 根天线
        initial_action = np.full(self.N_AP, 8, dtype=np.int32)
        self.prev_action = initial_action

        # 进行一次仿真
        SE, Ptot, _, CSI, SE_vec = self._simulate(initial_action)
        outage_cnt = int(np.sum(SE_vec < self.se_thr))
        reward = self._calc_reward(initial_action, outage_cnt)

        # 记录日志
        self.logger.info(f"[RESET] Action: {initial_action.tolist()}, SE: {SE:.4f}, "
                         f"Ptot: {Ptot:.2f}, Outage: {outage_cnt}, SE_vec: {np.array2string(SE_vec, precision=3)}")

        # 构造状态
        state = self._build_state(CSI, initial_action, SE, Ptot, reward, outage_cnt)
        return state, {}

    # ------------------------------------------------------ #
    # step()                                                 #
    # ------------------------------------------------------ #
    def step(self, action: np.ndarray):
        # Gym 可能传 list，这里确保是 ndarray
        action = np.asarray(action, dtype=np.int32)
        assert action.shape == (self.N_AP,), "Action shape mismatch"

        # 将离散动作 0/1/2 映射到 0/4/8 根天线
        action_mapped = action * 4   # 0->0,1->4,2->8

        # 仿真
        SE, Ptot, _, CSI, SE_vec = self._simulate(action_mapped)
        outage_cnt = int(np.sum(SE_vec < self.se_thr))
        reward = self._calc_reward(action_mapped, outage_cnt)

        # 构造状态
        state = self._build_state(CSI, action_mapped, SE, Ptot, reward, outage_cnt)

        # 更新缓存
        self.prev_action = action_mapped
        self.prev_SE     = SE
        self.prev_Ptot   = Ptot
        self.current_step += 1

        # 结束条件：仅使用时间截断 (TimeLimit)
        terminated = False
        truncated  = self.current_step >= self.max_steps
        info = {"SE": SE, "Ptot": Ptot, "Outage": outage_cnt}
        if truncated:
            info["TimeLimit.truncated"] = True

        # 日志
        self.logger.info(
            f"Step {self.current_step:03d} | SE {SE:.3f} | Ptot {Ptot:.1f} | Reward {reward:.3f} | "
            f"Outage {outage_cnt} | Active {int(action_mapped.sum())}")

        return state, reward, terminated, truncated, info

    # ------------------------------------------------------ #
    # 内部工具函数                                           #
    # ------------------------------------------------------ #
    def _build_state(self,
                     CSI: np.ndarray,
                     action_mapped: np.ndarray,
                     SE: float,
                     Ptot: float,
                     reward: float,
                     outage_cnt: int) -> np.ndarray:
        """将 CSI+动作+标量拼接为一维观测向量并做归一化"""
        CSI_norm = self._normalize_CSI(CSI)
        state = np.concatenate([
            CSI_norm.flatten(),          # (N_AP*N_UE,)
            action_mapped / 8.0,         # 0/4/8 -> 0/0.5/1
            np.array([
                SE / 12.0,              # 经验上 SE<=12
                Ptot / 2600.0,          # 经验上 Ptot<=2.6 kW
                reward,                 # 已在 [-2,0] 附近
                outage_cnt / self.N_UE
            ], dtype=np.float32)
        ]).astype(np.float32)
        return state

    def _calc_reward(self, action_mapped: np.ndarray, outage_cnt: int) -> float:
        """soft‑penalty 节能 + outage penalty"""
        active = float(action_mapped.sum())
        norm_active = active / (self.N_AP * 8)             # ∈ [0,1]
        energy_pen  = norm_active ** 1.5                   # 加重高功耗区间
        reward = - energy_pen - outage_cnt / self.N_UE
        return reward

    def _simulate(self, action_mapped: np.ndarray) -> Tuple[float, float, Any, np.ndarray, np.ndarray]:
        """调用 MATLAB `simulateConfig` 函数。"""
        try:
            N_ap_matlab = matlab.double(action_mapped.reshape(-1, 1).tolist())
            SE_vec, SE_mean, Ptot, avg_N_ap, gainOverNoisedB = self.eng.simulateConfig(
                N_ap_matlab,
                float(self.N_AP),
                float(self.N_UE),
                float(self.seed),
                nargout=5)
            SE_vec_np = np.array(SE_vec).flatten()
            CSI_np    = np.array(gainOverNoisedB, dtype=np.float32)
            return float(SE_mean), float(Ptot), avg_N_ap, CSI_np, SE_vec_np
        except matlab.engine.MatlabExecutionError as e:
            self.logger.error(f"MATLAB error: {e}")
            raise

    @staticmethod
    def _normalize_CSI(CSI: np.ndarray) -> np.ndarray:
        mean = CSI.mean()
        std  = CSI.std() + 1e-8
        return (CSI - mean) / std

    def close(self):
        self.eng.quit()