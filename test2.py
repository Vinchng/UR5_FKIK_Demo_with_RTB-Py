# tests/test_ik_analytic_roundtrip.py
import numpy as np
from spatialmath import SE3
from kinematics.ur5_dh import make_ur5
from kinematics.ik_analytic_ur5 import ik_ur5_analytic
from kinematics.ik_analytic_ur5 import ik_ur5_analytic_wrt_robot
# ---------- 小工具 ----------
def wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2*np.pi) - np.pi

def angle_l2(q1, q2):
    dq = wrap_to_pi(q1) - wrap_to_pi(q2)
    return float(np.linalg.norm(dq))

def fk_pose_err(robot, T_target, q):
    Terr = T_target.inv() * robot.fkine(q)
    perr = float(np.linalg.norm(Terr.t))
    ang  = float(np.arccos(np.clip((np.trace(Terr.R)-1)/2, -1, 1)))
    return perr, np.degrees(ang)

# ---------- 单例回环测试 ----------
def test_one(robot, q_ref, tol_rad=1e-5):
    q_ref = np.asarray(q_ref, dtype=float).reshape(6)
    T = robot.fkine(q_ref)  # 这是 TCP 位姿
    # 用包装器：先转到 D-H 法兰，再解
    q_best, sols = ik_ur5_analytic_wrt_robot(robot, T, q_previous=q_ref)
    err = angle_l2(q_best, q_ref)

    print("\n=== Case ===")
    print("q_ref (deg)  :", np.round(np.rad2deg(q_ref), 3))
    print("q_best (deg) :", np.round(np.rad2deg(q_best), 3))
    print("||Δq|| (rad) :", err)

    # 回代误差（用 q_best）
    perr, ang = fk_pose_err(robot, T, q_best)
    print(f"FK residual (best) : pos={perr:.3e} m, ang={ang:.3e} deg")

    if err <= tol_rad:
        print("✅ PASS: 解析IK回到同一分支（与 seed 一致）")
        return True

    # 如果没有回到同一分支，找 8 组解里离 q_ref 最近的一组，看看差距
    dists = [angle_l2(s, q_ref) for s in sols]
    idx = int(np.argmin(dists))
    q_near = sols[idx]
    perrN, angN = fk_pose_err(robot, T, q_near)

    print("⚠️ 没有回到完全相同的分支，展示 8 解中最近的一组：")
    print("idx_near      :", idx)
    print("q_near (deg)  :", np.round(np.rad2deg(q_near), 3))
    print("||Δq_near||   :", dists[idx])
    print(f"FK residual(nearest): pos={perrN:.3e} m, ang={angN:.3e} deg")

    # 打印所有 8 组（可注释）
    print("\n-- All 8 solutions (deg) --")
    for i, s in enumerate(sols):
        print(f"{i}: {np.round(np.rad2deg(s),3)}  ||Δq||={angle_l2(s,q_ref):.4e}")

    return False

if __name__ == "__main__":
    robot = make_ur5()  # 注意：这里应已设置好 ur5.tool 外参

    # 你可以在这里放多组要测试的 q
    cases_deg = [
        [0, -90, 90, 0, 90, 0],
        [30, -60, 80, 10, 70, -20],
        [-45, -100, 110, 20, 60, 30],
    ]
    cases = [np.deg2rad(c) for c in cases_deg]

    all_ok = True
    for q in cases:
        ok = test_one(robot, q, tol_rad=1e-5)
        all_ok = all_ok and ok

    print("\n========== Summary ==========")
    print("All cases pass same-branch round-trip:", all_ok)
