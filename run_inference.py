import logging
import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from environment import CellFreeMiMoCSIEnv

# ---------- 1. 载入模型 ----------
model_path = os.path.join("ppo_ckpt", "4ue-se3.zip")
model = PPO.load(model_path, device="cpu")

# ---------- 2. 创建环境 ----------
env = CellFreeMiMoCSIEnv(
    N_AP=16,
    N_UE=4,
    max_steps=256,
    se_threshold=1.0
)

log_time = datetime.now().strftime("%Y%m%d-%H%M%S")
infer_log_dir = os.path.join("./inference_logs", log_time)
os.makedirs(infer_log_dir, exist_ok=True)

fh = logging.FileHandler(os.path.join(infer_log_dir, "inference_steps.txt"))
fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
env.logger.addHandler(fh)

# 总览 logger
summary_logger = logging.getLogger("InferenceSummary")
summary_logger.setLevel(logging.INFO)
sh = logging.FileHandler(os.path.join(infer_log_dir, "summary.txt"))
sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
summary_logger.addHandler(sh)

# ---------- 3. 推理 ----------
seed = 93
state, _ = env.reset(seed=seed)

# 【这里改成按 outage 记录】
best_per_outage = {}  # 例如 {0: {...}, 1: {...}, 2: {...}, 3: {...}, 4: {...}}

done = False
while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    outage = info["Outage"]
    ptot = info["Ptot"]

    # 只考虑 outage 0-4
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

# ───────────────────── 4. 记录 summary ───────────────────── #
for outage_level in range(0, 5):  # 只输出 0-4
    if outage_level in best_per_outage:
        best = best_per_outage[outage_level]
        msg = (f"[Outage {outage_level}] BEST STEP {best['step']:03d} | Antennas {best['antennas'].tolist()} | "
               f"SE {best['se']:.4f} | Ptot {best['ptot']:.2f}")
        print(msg)
        summary_logger.info(msg)
    else:
        msg = f"[Outage {outage_level}] No available step."
        print(msg)
        summary_logger.info(msg)

env.close()
