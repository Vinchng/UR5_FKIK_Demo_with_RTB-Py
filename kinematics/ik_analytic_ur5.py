# kinematics/ik_analytic_ur5.py
# Port of "ikSolverUR5All" (Rasmus Skovgaard Andersen UR5 kinematics) from MATLAB to Python.
# Requires: numpy, spatialmath (for SE3) optional; but here we only need numpy.

import numpy as np
from spatialmath import SE3

# ---- UR5 DH parameters (UR official) ----
_a = np.array([0, -0.425, -0.39225, 0, 0, 0], dtype=float)
_d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=float)
_alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0], dtype=float)

_EPS = 1e-9

def ik_ur5_analytic_wrt_robot(robot, T_target: SE3, q_previous=None):
    """
    解析 IK（相对 robot 的 base/tool）：
      输入:  robot, 目标 TCP 位姿 T_target (SE3)，上一姿态 q_previous (rad)
      内部:  计算 T06 = base^{-1} * T_target * tool^{-1}，再调用核心解析器
      返回:  (q_best, sols)  同核心版
    """
    # 统一成 SE3
    if not isinstance(T_target, SE3):
        T_target = SE3(T_target)

    # 取 base/tool（若未设置则为单位阵）
    B = robot.base if isinstance(robot.base, SE3) else SE3(robot.base)
    X = robot.tool if isinstance(robot.tool, SE3) else SE3(robot.tool)

    # 目标从 TCP 转到 D-H 法兰（link-6）坐标系
    T06 = B.inv() * T_target * X.inv()

    # 调用核心解析器（注意传 numpy 4x4）
    from .ik_analytic_ur5 import ik_ur5_analytic as _core
    q_best, sols = _core(T06.A, q_previous=q_previous)
    return q_best, sols

def _clamp(x, lo=-1.0, hi=1.0):
    return float(np.clip(x, lo, hi))

def _DH2tform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0,0] = ct;     T[0,1] = -st;     T[0,2] = 0;   T[0,3] = a
    T[1,0] = st*ca;  T[1,1] = ct*ca;   T[1,2] = -sa; T[1,3] = -sa*d
    T[2,0] = st*sa;  T[2,1] = ct*sa;   T[2,2] = ca;  T[2,3] =  ca*d
    return T

def _calculateTheta6(X60, Y60, th1, th5):
    # X60, Y60: columns of R60 (3,)
    if abs(np.sin(th5)) < _EPS:
        return 0.0
    left  = (-X60[1]*np.sin(th1) + Y60[1]*np.cos(th1)) / np.sin(th5)
    right = ( X60[0]*np.sin(th1) - Y60[0]*np.cos(th1)) / np.sin(th5)
    return np.arctan2(left, right)

def _calculateP14(T06, th1, th5, th6):
    T01 = _DH2tform(0.0, 0.0, _d[0], th1)
    T10 = np.linalg.inv(T01)

    T45 = _DH2tform(_alpha[3], _a[3], _d[4], th5)  # (alpha4, a4, d5, th5)
    T54 = np.linalg.inv(T45)

    T56 = _DH2tform(_alpha[4], _a[4], _d[5], th6)  # (alpha5, a5, d6, th6)
    T65 = np.linalg.inv(T56)

    T14 = T10 @ T06 @ T65 @ T54
    P14 = T14[0:3, 3]
    return P14, T14

def _calculateTheta3(T06, th1, th5, th6):
    P14, T14 = _calculateP14(T06, th1, th5, th6)
    Lxz = np.linalg.norm([P14[0], P14[2]])
    # conditions: |a2 - a3| < Lxz < |a2 + a3|
    c1 = abs(_a[1] - _a[2]); c2 = abs(_a[1] + _a[2])
    if not (Lxz > c1 - 1e-12 and Lxz < c2 + 1e-12):
        raise ValueError("Theta3 cannot be determined: geometry out of range")
    cos_th3 = _clamp((Lxz**2 - _a[1]**2 - _a[2]**2) / (2*_a[1]*_a[2]))
    th3 = np.arccos(cos_th3)
    return th3, P14, T14

def _rearange(idx):
    # MATLAB reindex: [1 3 2 4 5 7 6 8] -> zero-based [0,2,1,3,4,6,5,7]
    return [0,2,1,3,4,6,5,7][idx]

def _generate_possible_solutions(theta1, theta2, theta3, theta4, theta5, theta6):
    # Follow the same pattern as MATLAB generatePossibleSolutions()
    sols = np.zeros((8,6), dtype=float)
    sols[0,:] = [theta1[0], theta2[0], theta3[0], theta4[0], theta5[0], theta6[0]]
    sols[1,:] = [theta1[0], theta2[2], theta3[2], theta4[2], theta5[1], theta6[1]]
    sols[2,:] = [theta1[1], theta2[4], theta3[4], theta4[4], theta5[2], theta6[2]]
    sols[3,:] = [theta1[1], theta2[6], theta3[6], theta4[6], theta5[3], theta6[3]]

    sols[4,:] = [theta1[0], theta2[1], theta3[1], theta4[1], theta5[0], theta6[0]]
    sols[5,:] = [theta1[0], theta2[3], theta3[3], theta4[3], theta5[1], theta6[1]]
    sols[6,:] = [theta1[1], theta2[5], theta3[5], theta4[5], theta5[2], theta6[2]]
    sols[7,:] = [theta1[1], theta2[7], theta3[7], theta4[7], theta5[3], theta6[3]]
    return sols

def _closest_solution(solutions, q_prev):
    w = np.array([6,5,4,3,2,1], dtype=float)
    best_i = 0; best_cost = np.inf
    for i in range(solutions.shape[0]):
        dq = (solutions[i,:] - q_prev) * w
        cost = float(np.sum(dq*dq))
        if cost < best_cost:
            best_cost, best_i = cost, i
    return solutions[best_i,:]

def ik_ur5_analytic(T06: np.ndarray, q_previous=None):
    """
    解析 IK（Rasmus 方案，返回“最接近上一姿态”的一组解）：
      T06: 4x4 末端位姿（基于与本模块相同的 UR5 DH 定义）
      q_previous: 6x (rad)，若 None 则全 0
    返回: (q_best (6,), solutions_all (8,6))
    可能抛出 ValueError 表示几何不可达（腕/肘约束破坏）
    """
    if q_previous is None:
        q_previous = np.zeros(6)
    q_previous = np.asarray(q_previous, dtype=float).reshape(6)

    # ---------- θ1 (2 solutions) ----------
    P06 = T06[0:3, 3]
    P05_h = T06 @ np.array([0,0,-_d[5],1.0])  # [0;0;-d6;1]
    P05 = P05_h[0:3]

    th1P = np.arctan2(P05[1], P05[0]) + np.pi/2
    th1M = th1P
    r = np.hypot(P05[0], P05[1])
    if r > _EPS:
        phi = np.arccos(_clamp(_d[3]/r))
        th1P = th1P + phi
        th1M = th1M - phi
    theta1 = np.array([th1P, th1M])

    # ---------- θ5 (2 for each θ1) -> 4 ----------
    theta5 = np.zeros(4)
    k = 0
    for th1 in theta1:
        acos_val = (P06[0]*np.sin(th1) - P06[1]*np.cos(th1) - _d[3]) / _d[5]
        acos_val = _clamp(acos_val)
        for sgn in (1, -1):
            theta5[k] = sgn * np.arccos(acos_val)
            k += 1

    # ---------- θ6 (1 for each θ1,θ5) -> 4 ----------
    R60 = np.linalg.inv(T06)[0:3, 0:3]
    X60 = R60[:,0]; Y60 = R60[:,1]
    theta6 = np.zeros(4)
    idx5 = [0, 2]  # pair indices for θ5 that correspond to θ1[0], θ1[1]
    k = 0
    for i_th1 in range(2):
        th1 = theta1[i_th1]
        theta6[k]   = _calculateTheta6(X60, Y60, th1, theta5[idx5[i_th1]])
        theta6[k+1] = _calculateTheta6(X60, Y60, th1, theta5[idx5[i_th1]+1])
        k += 2

    # ---------- θ3 (2 sign branches × 4 combos) -> 8 ----------
    theta3 = np.zeros(8); P14_list = np.zeros((8,3)); T14_list = np.zeros((8,4,4))
    k = 0
    # follow MATLAB loop structure to pair (θ1,θ5,θ6)
    t5_pairs = [0, 2]
    for i_th1 in range(2):
        th1 = theta1[i_th1]
        for sgn in (1, -1):
            # with θ5[paired], θ6[paired]
            th5 = theta5[t5_pairs[i_th1]]
            th6 = theta6[t5_pairs[i_th1]]
            th3, P14, T14 = _calculateTheta3(T06, th1, th5, th6)
            theta3[k]   =  sgn * th3; P14_list[k,:] = P14; T14_list[k,:,:] = T14; k += 1

            th5 = theta5[t5_pairs[i_th1]+1]
            th6 = theta6[t5_pairs[i_th1]+1]
            th3, P14, T14 = _calculateTheta3(T06, th1, th5, th6)
            theta3[k]   =  sgn * th3; P14_list[k,:] = P14; T14_list[k,:,:] = T14; k += 1

    # Rearrange order to match MATLAB pairing
    order = [0,2,1,3,4,6,5,7]
    theta3 = theta3[order]; P14_list = P14_list[order,:]; T14_list = T14_list[order,:,:]

    # ---------- θ2 (1 per each) -> 8 ----------
    theta2 = np.zeros(8)
    for i in range(8):
        P14 = P14_list[i,:]
        Lxz = np.linalg.norm([P14[0], P14[2]])
        # atan2 uses (-z,-x) as in MATLAB
        theta2[i] = np.arctan2(-P14[2], -P14[0]) - np.arcsin(-_a[2]*np.sin(theta3[i]) / max(Lxz, _EPS))

    # ---------- θ4 (1 per each) -> 8 ----------
    theta4 = np.zeros(8)
    for i in range(8):
        T14 = T14_list[i,:,:]
        # T12(θ2), T23(θ3)
        T12 = _DH2tform(_alpha[0], _a[0], _d[1], theta2[i])
        T21 = np.linalg.inv(T12)
        T23 = _DH2tform(_alpha[1], _a[1], _d[2], theta3[i])
        T32 = np.linalg.inv(T23)
        T34 = T32 @ T21 @ T14
        X34 = T34[0:3, 0]
        theta4[i] = np.arctan2(X34[1], X34[0])

    # ---------- collect 8 solutions & choose closest ----------
    # pair θ1 (2), θ5(4), θ6(4) already consistent with MATLAB generatePossibleSolutions
    th1 = theta1
    th5 = theta5
    th6 = theta6
    sols = _generate_possible_solutions(th1, theta2, theta3, theta4, th5, th6)

    # wrap to [-pi, pi] for neatness
    sols = (sols + np.pi) % (2*np.pi) - np.pi
    q_prev = (q_previous + np.pi) % (2*np.pi) - np.pi

    q_best = _closest_solution(sols, q_prev)
    return q_best, sols
