# kinematics/ik_numeric.py
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb

def wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2*np.pi) - np.pi

def _solve_once(robot, T: SE3, q0, solver="LM", tol=1e-6, ilimit=200, slimit=200):
    if solver.upper() == "LM":
        sol = robot.ikine_LM(T, q0=q0, tol=tol, ilimit=ilimit, slimit=slimit)
    else:
        sol = robot.ikine_NR(T, q0=q0, tol=tol, ilimit=ilimit, slimit=slimit)
    ok = bool(sol.success)
    q  = wrap_to_pi(sol.q) if sol.q is not None else None
    # 回代残差（位置+姿态角）
    if ok and q is not None:
        Terr = T.inv() * robot.fkine(q)
        perr = np.linalg.norm(Terr.t)
        ang  = np.arccos(np.clip((np.trace(Terr.R) - 1)/2, -1, 1))
        resid = float(perr + ang)   # 简单标量化
    else:
        resid = np.inf
    info = {
        "success": ok,
        "reason": getattr(sol, "reason", ""),
        "iters": getattr(sol, "iterations", None),
        "resid": getattr(sol, "residual", None),
        "perr": perr if ok else None,
        "ang":  np.degrees(ang) if ok else None,
    }
    return ok, q, resid, info

def ik_numeric(robot, T: SE3, q0=None, seeds_extra=None,
               solver="LM", tol=1e-6, ilimit=200, slimit=200,
               n_random=8, rand_scale_deg=25.0):
    """
    多初值数值 IK（挑选最佳解）:
      - q0: 基础初值（若 None 默认全零）
      - seeds_extra: 额外候选种子（弧度），例如来自用户种子
      - n_random: 随机扰动种子数量
    返回: (success, q_best, best_info)
    """
    n = robot.n
    if q0 is None:
        q0 = np.zeros(n)

    seeds = [wrap_to_pi(q0)]

    # 翻转组合（常见分支）
    flips = [
        np.r_[0,0,0, 0,0,0],
        np.r_[0,0,0, np.pi,0,0],
        np.r_[0,0,0, 0,0,np.pi],
        np.r_[0,0,0, np.pi,0,np.pi],
    ]
    seeds.extend([wrap_to_pi(q0 + f) for f in flips])

    # 用户额外种子
    if seeds_extra is not None:
        for s in np.atleast_2d(seeds_extra):
            seeds.append(wrap_to_pi(s))

    # 随机扰动几组
    if n_random > 0:
        scale = np.deg2rad(rand_scale_deg)
        for _ in range(n_random):
            seeds.append(wrap_to_pi(q0 + np.random.uniform(-scale, scale, size=n)))

    # 逐个尝试，选 resid 最小的
    best = (np.inf, None, None)  # (resid, q, info)
    for s in seeds:
        ok, q, resid, info = _solve_once(robot, T, q0=s, solver=solver,
                                         tol=tol, ilimit=ilimit, slimit=slimit)
        if ok and resid < best[0]:
            best = (resid, q, info)

    success = best[1] is not None
    return success, best[1], best[2] if success else {"success": False, "reason": "no converged seed"}
