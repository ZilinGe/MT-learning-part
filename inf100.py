import os
import re
import logging
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from environment import CellFreeMiMoCSIEnv

# ========== 配置 ==========
BASE_DIR = r"C:\Users\asus\Desktop\master\MasterThesis_learningpart\ppo_ckpt\result"
N_AP = 16
N_RUNS = 100  # 每个模型推理次数

# ========== 工具函数 ==========

def find_models(base_dir):
    model_info = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".zip"):
                    zip_path = os.path.join(folder_path, file)
                    model_info.append((folder, zip_path))
    return model_info

def parse_folder_name(name):
    match = re.match(r"(\d+)ue-se(\d+)", name)
    if match:
        return int(match.group(1)), float(match.group(2))
    else:
        raise ValueError(f"文件夹命名格式错误：{name}（应为例如 4ue-se3）")

def run_multiple_inference(model_path, N_UE, se_threshold, log_dir):
    env = CellFreeMiMoCSIEnv(
        N_AP=N_AP,
        N_UE=N_UE,
        max_steps=256,
        se_threshold=se_threshold
    )

    os.makedirs(log_dir, exist_ok=True)

    # 设置日志
    fh = logging.FileHandler(os.path.join(log_dir, "inference_steps.txt"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    env.logger.addHandler(fh)

    summary_logger = logging.getLogger(f"Summary_{N_UE}_{se_threshold}")
    summary_logger.setLevel(logging.INFO)
    sh = logging.FileHandler(os.path.join(log_dir, "summary.txt"))
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    summary_logger.addHandler(sh)

    # 加载模型
    model = PPO.load(model_path, device="cpu")

    ptot_records = {i: [] for i in range(5)}

    for run_id in range(N_RUNS):
        print(f"▶️  Running {run_id + 1}/{N_RUNS} ...")
        seed = np.random.randint(0, 100000)
        state, _ = env.reset(seed=seed)

        best_per_outage = {}
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            outage = info["Outage"]
            ptot = info["Ptot"]

            if 0 <= outage <= 4:
                if (outage not in best_per_outage) or (ptot < best_per_outage[outage]["ptot"]):
                    best_per_outage[outage] = {
                        "step": env.current_step,
                        "action": action.copy(),
                        "antennas": env.prev_action.copy(),
                        "outage": outage,
                        "ptot": ptot,
                        "se": info["SE"],
                    }

        for outage_level in range(5):
            if outage_level in best_per_outage:
                ptot_records[outage_level].append(best_per_outage[outage_level]["ptot"])

    env.close()

    # 总结输出
    print("\n────────── SUMMARY ──────────")
    summary_logger.info("────────── SUMMARY ──────────")

    for outage_level in range(5):
        samples = ptot_records[outage_level]
        if samples:
            avg_ptot = np.mean(samples)
            msg = f"[Outage {outage_level}] Samples: {len(samples)} | Average Ptot: {avg_ptot:.2f}"
        else:
            msg = f"[Outage {outage_level}] No valid samples."
        print(msg)
        summary_logger.info(msg)

# ========== 主程序：批量处理每个模型 ==========

if __name__ == "__main__":
    model_list = find_models(BASE_DIR)
    inference_root_dir = os.path.join(os.getcwd(), "inference_logs")
    os.makedirs(inference_root_dir, exist_ok=True)

    for folder_name, model_path in model_list:
        try:
            print(f"\n\n💼 开始处理模型：{folder_name}")
            N_UE, se_thr = parse_folder_name(folder_name)
            # 日志目录
            log_dir = os.path.join(inference_root_dir, folder_name)

            run_multiple_inference(model_path, N_UE, se_thr, log_dir)
        except Exception as e:
            print(f"❌ 处理失败：{folder_name}：{e}")
