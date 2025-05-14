import matlab.engine
import numpy as np

# start MATLAB eng
eng = matlab.engine.start_matlab()

# Path of
eng.addpath(eng.genpath(r'C:\Users\asus\Desktop\master\003-master_thesis\002 - 004_dnn'), nargout=0)

# N_ap arry
N_ap = np.ones((16, 1)) * 8

# Trans N_ap to MATLAB Arry
N_ap_matlab = matlab.double(N_ap.tolist())  # 将 NumPy 数组转换为 MATLAB 数组
print(N_ap_matlab)

# Call from matlab
# SE_results, Ptot_results, avg_N_ap = eng.simulateConfig(N_ap_matlab, nargout=3)


N_AP = 16
N_UE = 4
Seed = 123

SE_results, SE_mean, Ptot_results, avg_N_ap, gainOverNoisedB =eng.simulateConfig(
            N_ap_matlab, matlab.double(N_AP), matlab.double(N_UE),matlab.double(Seed), nargout=5
        )


# 打印输出结果
print(SE_results)
print(Ptot_results)
print(avg_N_ap)

# 关闭 MATLAB 引擎
eng.quit()