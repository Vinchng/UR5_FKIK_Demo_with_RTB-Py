import numpy as np
from spatialmath import SE3
from kinematics.ur5_dh import make_ur5

# 1) 你的那组关节角（度）
q_deg = [0, -90, 90, 0, 90, 0]
q = np.deg2rad(q_deg)

# 2) 你的“期望”TCP矩阵（world->TCP）
T_target = SE3([[ 0.0, -1.0,  0.0,  0.3],
                [ 1.0,  0.0,  0.0,  0.2],
                [ 0.0,  0.0,  1.0,  0.5],
                [ 0.0,  0.0,  0.0,  1.0]])

# 3) 取 D-H 机器人、算当前的 DH 末端（world->link6）
robot = make_ur5()
T_dh = robot.fkine(q)      # SE3

# 4) 求出固定工具外参 X（link6->TCP）
X_tool = T_dh.inv() * T_target

# 5) 写回：让以后一切都在你的 TCP 下工作
robot.tool = X_tool

# 6) 验证：现在再算 FK 应 ≈ 你的 T_target
T_check = robot.fkine(q)
print("X_tool =\n", X_tool.A)
print("T_check (should match T_target) =\n", T_check.A)
print("ΔT =\n", (T_check.inv() * T_target).A)  # 应接近单位阵
