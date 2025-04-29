import logging
import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from environment import CellFreeMiMoCSIEnv

# ---------- 1. 载入模型 ----------
model_path = os.path.join("ppo_ckpt", "ppo_step_300000.zip")
model = PPO.load(model_path, device="cpu")

# ---------- 2. 创建环境 ----------
env = CellFreeMiMoCSIEnv(
    N_AP=16,
    N_UE=4,
    max_steps=256,
    se_threshold=3.0
)

log_time = datetime.now().strftime("%Y%m%d-%H%M%S")
infer_log_dir = os.path.join("./inference_logs", log_time)
os.makedirs(infer_log_dir, exist_ok=True)

# 推理细节 logger
fh = logging.FileHandler(os.path.join(infer_log_dir, "inference_steps.txt"))
fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
env.logger.addHandler(fh)

# 总结 logger
summary_logger = logging.getLogger("InferenceSummary")
summary_logger.setLevel(logging.INFO)
sh = logging.FileHandler(os.path.join(infer_log_dir, "summary.txt"))
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
summary_logger.addHandler(sh)

# ---------- 3. 推理多次 ----------
N_runs = 100  # 运行次数
ptot_records = {i: [] for i in range(5)}  # 记录每个outage级别的ptot列表

for run_id in range(N_runs):
    print(f"Running {run_id + 1}/{N_runs} ...")  # <-- 新增这行，run_id从0开始所以+1

    seed = np.random.randint(0, 100000)  # 每次随机一个seed
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

    # 推理结束后，统计ptot
    for outage_level in range(0, 5):
        if outage_level in best_per_outage:
            ptot_records[outage_level].append(best_per_outage[outage_level]["ptot"])

env.close()

# ---------- 4. 总结 ----------
print("\n\n────────── SUMMARY ──────────")
summary_logger.info("────────── SUMMARY ──────────")

for outage_level in range(0, 5):
    samples = ptot_records[outage_level]
    if len(samples) > 0:
        avg_ptot = np.mean(samples)
        msg = (f"[Outage {outage_level}] Samples: {len(samples)} | Average Ptot: {avg_ptot:.2f}")
    else:
        msg = (f"[Outage {outage_level}] No valid samples.")
    print(msg)
    summary_logger.info(msg)
