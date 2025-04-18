import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matlab.engine
import os
import logging
import time
from datetime import datetime

class CellFreeMiMoCSIEnv(gym.Env):
    def __init__(self, N_AP, N_UE, seed=123):
        super().__init__()

        # AP 和 UE 数量
        self.N_AP = N_AP
        self.N_UE = N_UE

        self.seed = seed

        # 初始化 logger
        self.logger = logging.getLogger("CellFreeMiMoCSIEnvLogger")
        self.logger.setLevel(logging.INFO)

        # 创建 handler，只创建一次避免重复写入
        if not self.logger.handlers:
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = f"./env_logs/{current_time}"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "env_step_log.txt")
            file_handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        print(f"Initial self.N_AP: {self.N_AP}, self.N_UE: {self.N_UE}")

        # 启动 MATLAB 引擎
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(r'C:\Users\asus\Desktop\master\003-master_thesis\002 - 004_dnn'), nargout=0)
        print("[MATLAB PATH] 当前 MATLAB 路径设置为:")
        print(self.eng.path())

        # 动作空间定义：每个 AP 可以选择的天线数量 (0, 4, 8)
        self.action_space = spaces.MultiDiscrete([3] * self.N_AP)  # 0: 0 antennas, 1: 4 antennas, 2: 8 antennas

        # 状态空间定义：CSI 矩阵 (AP x UE) + 上一步动作 (AP) + 额外特征 (4)
        state_size = N_AP * N_UE + N_AP + 4
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(state_size,), dtype=np.float32)

        # 奖励权重
        self.w_SE = 1.0
        self.w_Ptot = 0.005

        # 上一步信息（初始化）
        self.prev_action = np.zeros(self.N_AP)
        self.prev_SE = 0.0
        self.prev_Ptot = 0.0

    def reset(self, seed=None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed = seed

        # 初始化全部为 8 根天线
        initial_actions = np.full(self.N_AP, 8)
        self.prev_action = initial_actions

        # 调用 MATLAB 仿真，获取初始 CSI、SE、Ptot
        SE, Ptot, _, CSI, SE_results_vec = self._simulate(initial_actions)

        # 保存历史
        self.prev_SE = SE
        self.prev_Ptot = Ptot

        # 计算 outage count
        threshold = 7
        outage_count = np.sum(SE_results_vec < threshold)

        # 估算初始 reward（与 step 中一致）
        total_active_antennas = np.sum(initial_actions)
        reward = - total_active_antennas / (self.N_AP * 8) - outage_count * 1.0

        # 日志记录
        SE_results_str = np.array2string(SE_results_vec, precision=3, separator=', ')
        self.logger.info(f"[RESET] Init Action: {self.prev_action.tolist()}, SE: {SE:.4f}, "
                         f"Ptot: {Ptot:.2f}, SE_results: {SE_results_str}")

        # 构造状态（与 step 一致）
        CSI_norm = self._normalize_CSI(CSI)
        state = np.concatenate([
            CSI_norm.flatten(),
            self.prev_action / 16.0,
            np.array([
                SE / 10.0,  # 归一化 SE
                Ptot / 2600.0,  # 归一化 Ptot
                reward,  # 当前 reward
                outage_count / self.N_UE  # 归一化 outage count
            ])
        ])

        return state, {}

    def step(self, action):
        # 确保action总是作为数组处理，即使它是单个数字
        if not isinstance(action, np.ndarray):
            action = np.array([action])  # 将单个动作转换为数组

        # 根据动作映射到新的动作值
        # action_mapped = np.array([0 if x == 0 else 8 for x in action])
        action_delta = np.array([-1 if x == 0 else 0 if x == 1 else 1 for x in action])
        # print(action_mapped)

        new_action = np.clip(self.prev_action + action_delta, 0, 8)

        # 调用 MATLAB 仿真
        avg_SE, Ptot, _, CSI, SE_results_vec = self._simulate(new_action)

        # 归一化总功率
        norm_Ptot = (Ptot - 0) / (2600 - 0)

        # 计算中断数
        # 3
        threshold = 7
        outage_count = np.sum(SE_results_vec < threshold)

        # 计算奖励
        weight = 0.6  # 惩罚系数
        k = self.N_UE  # 可以调整为其他合适的归一化因子

        # 统计总开通天线数（例如：0,2,4,6,8）
        total_active_antennas = np.sum(new_action)

        # 统计多少个 UE 的 SE 低于阈值
        outage_count = np.sum(SE_results_vec < threshold)

        reward = - total_active_antennas / (self.N_AP * 8) - outage_count * 1.0

        # 日志记录
        SE_results_str = np.array2string(SE_results_vec, precision=3, separator=', ')
        self.logger.info(
            f"Action Delta: {action_delta.tolist()}, New Action: {new_action.tolist()}, "
            f"SE: {avg_SE:.4f}, Ptot: {Ptot:.2f}, Reward: {reward:.4f}, "
            f"Outage Count: {outage_count}, Active Antennas: {total_active_antennas}, "
            f"SE_results: {SE_results_str}"
        )

        # 更新状态（与 reset 保持一致结构）
        CSI_norm = self._normalize_CSI(CSI)
        state = np.concatenate([
            CSI_norm.flatten(),
            new_action / 16.0,
            np.array([
                avg_SE / 10.0,  # 归一化 SE
                Ptot / 2600.0,  # 归一化 Ptot
                reward,  # 当前 reward
                outage_count / self.N_UE  # 归一化 outage count
            ])
        ])

        # self.prev_action = action_mapped
        self.prev_action = new_action
        self.prev_SE = avg_SE
        self.prev_Ptot = Ptot

        terminated = False
        truncated = False
        info = {'SE': avg_SE, 'Ptot': Ptot, 'Outage Count': outage_count}

        return state, reward, terminated, truncated, info

    def _simulate(self, action):
        """调用 MATLAB 进行仿真，并打印输入参数以供检查。"""
        try:
            # 将动作转换为MATLAB双精度类型
            # N_ap_matlab = matlab.double(action.reshape(-1, 1).tolist())
            N_ap_matlab = matlab.double(action.reshape(-1, 1).tolist())
            # print(f"Calling MATLAB simulateConfig with: N_AP={N_ap_matlab}")

            # 打印传递给MATLAB函数的参数
            # print(f"Calling MATLAB simulateConfig with: N_AP={self.N_AP}, N_UE={self.N_UE}, Action={action.tolist()}")

            # 调用MATLAB函数
            SE_results, SE_mean, Ptot_results, avg_N_ap, gainOverNoisedB = self.eng.simulateConfig(
                N_ap_matlab,
                matlab.double(self.N_AP),
                matlab.double(self.N_UE),
                matlab.double(self.seed),
                nargout=5
            )

            # 将MATLAB返回的结果转换为Python可用的格式
            SE_results_np = np.array(SE_results).flatten()
            SE = float(SE_mean)
            Ptot = float(Ptot_results)
            CSI = np.array(gainOverNoisedB, dtype=np.float32)

            return SE, Ptot, avg_N_ap, CSI, SE_results_np
        except matlab.engine.MatlabExecutionError as e:
            print(f"MATLAB执行错误：{str(e)}")
            # 可以在这里添加额外的错误处理逻辑
            raise

    def _normalize_CSI(self, CSI):
        """ CSI 数据标准化，防止数值不稳定 """
        CSI_mean = np.mean(CSI)
        CSI_std = np.std(CSI) + 1e-8
        CSI_norm = (CSI - CSI_mean) / CSI_std
        return CSI_norm

    def close(self):
        self.eng.quit()

