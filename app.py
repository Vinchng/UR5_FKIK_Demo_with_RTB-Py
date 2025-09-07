# app.py
import argparse
import ast
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3, SO3
from kinematics.ur5_dh import make_ur5
from kinematics.ik_numeric import ik_numeric
from kinematics.ik_analytic_ur5 import ik_ur5_analytic_wrt_robot
from viz.mpl_view import plot_stick

# ---------- 解析工具 ----------
def parse_q(text, deg=True):
    text = text.strip()
    try:
        val = ast.literal_eval(text)
        arr = np.array(val, dtype=float).reshape(-1)
    except Exception:
        parts = [p for p in text.replace(",", " ").split() if p]
        arr = np.array([float(p) for p in parts], dtype=float)
    if arr.size != 6:
        raise ValueError(f"需要 6 个关节角，收到 {arr.size} 个。")
    return np.deg2rad(arr) if deg else arr

def parse_pose(line: str):
    s = line.strip()
    if s.startswith(("T=","t=")):
        M = ast.literal_eval(s.split("=",1)[1])
        M = np.array(M, dtype=float)
        if M.shape != (4,4):
            raise ValueError("T 需要是 4x4 矩阵")
        return SE3(M)

    kv = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = ast.literal_eval(v.strip())

    if "p" not in kv:
        try:
            p_try = ast.literal_eval(s)
            p_arr = np.array(p_try, dtype=float).reshape(3)
            return SE3(p_arr)
        except Exception:
            raise ValueError("缺少 p=[x,y,z] 或 T=[[...]]")

    p = np.array(kv["p"], dtype=float).reshape(3)

    if "rpy(rad)" in kv:
        rpy = np.array(kv["rpy(rad)"], dtype=float).reshape(3)
        R = SO3.RPY(rpy, order='xyz', unit='rad')
        return SE3.Rt(R, p)
    if "rpy" in kv:
        rpy = np.array(kv["rpy"], dtype=float).reshape(3)
        R = SO3.RPY(np.deg2rad(rpy), order='xyz', unit='rad')
        return SE3.Rt(R, p)
    if "quat" in kv:
        from spatialmath import UnitQuaternion
        qw, qx, qy, qz = [float(x) for x in kv["quat"]]
        R = UnitQuaternion([qw, qx, qy, qz]).SO3()
        return SE3.Rt(R, p)

    return SE3(p)

# ---------- 解析 IK 菜单辅助 ----------
def _wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2*np.pi) - np.pi

def _solution_menu_score(q, q_seed):
    w = np.array([6,5,4,3,2,1], dtype=float)
    dq = _wrap_to_pi(q - q_seed)
    return float(np.sum((dq*w)**2))

def _fk_residual(robot, T_target, q):
    Terr = T_target.inv() * robot.fkine(q)
    perr = float(np.linalg.norm(Terr.t))
    ang  = float(np.arccos(np.clip((np.trace(Terr.R)-1)/2, -1, 1)))
    return perr, np.degrees(ang)

def _print_solution_table(robot, T, sols, q_seed):
    print("\n[解析 IK] 候选 8 组解（按与 seed 距离排序）：")
    rows = []
    for i, q in enumerate(sols):
        perr, ang = _fk_residual(robot, T, q)
        score = _solution_menu_score(q, q_seed)
        rows.append((i, score, perr, ang, np.rad2deg(q)))
    rows.sort(key=lambda x: x[1])
    print(" idx | dist_to_seed | pos_err(m) | ang_err(deg) | q(deg)")
    for i, score, perr, ang, qdeg in rows:
        print(f"{i:>4} | {score:>13.4f} | {perr:>10.3e} | {ang:>10.3e} | {np.round(qdeg,2)}")
    return rows[0][0]

def _ik_solution_menu(robot, T, sols, q_seed, ax, lim=0.8):
    best_idx = _print_solution_table(robot, T, sols, q_seed)
    sel = best_idx
    q = sols[sel]
    plot_stick(robot, q, ax=ax, lim=lim, show_frames=True)
    print("\n操作：输入 0-7 选择；n 下一解；p 上一解；b 回最佳；q 退出菜单。")

    while True:
        cmd = input(f"[解析IK] 当前 idx={sel}，指令：").strip().lower()
        if cmd in ("q","quit","exit",""):
            break
        elif cmd == "n":
            sel = (sel + 1) % len(sols)
        elif cmd == "p":
            sel = (sel - 1) % len(sols)
        elif cmd == "b":
            sel = best_idx
        else:
            try:
                i = int(cmd)
                if 0 <= i < len(sols): sel = i
                else: print("索引范围 0~7"); continue
            except Exception:
                print("无效指令：0-7 / n / p / b / q"); continue

        q = sols[sel]
        plot_stick(robot, q, ax=ax, lim=lim, show_frames=True)
        perr, ang = _fk_residual(robot, T, q)
        print(f"→ idx={sel}: |pos|={perr:.3e} m, |ang|={ang:.3e} deg, q(deg)={np.round(np.rad2deg(q),2)}")

    return sols[sel]

# ---------- 交互会话 ----------
def interactive_fk(robot, lim=0.8):
    plt.ion(); fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    print("【FK 模式】\n  - 输入 6 个关节角（度）；示例：0 -90 90 0 90 0\n  - 'rad:' 前缀表示弧度；'quit' 退出。")
    while True:
        try:
            line = input("q = ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        if line.strip().lower() in ("q","quit","exit"): break
        try:
            q = parse_q(line.split(":",1)[1], deg=False) if line.strip().lower().startswith("rad:") \
                else parse_q(line, deg=True)
            T = robot.fkine(q)
            np.set_printoptions(precision=4, suppress=True)
            print("T =\n", T.A)
            plot_stick(robot, q, ax=ax, lim=lim, show_frames=True)
        except Exception as e:
            print("输入错误：", e)
    plt.ioff(); plt.show(); print("退出 FK。")

def interactive_ik(robot, lim=0.8, solver="numeric"):
    plt.ion(); fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    print(f"【IK 模式 | {solver}】\n  T=[[...]] 或 p=[x,y,z], rpy=[r,p,y] / rpy(rad)=[...] / quat=[qw,qx,qy,qz]\n  可选 seed=[...] (度) ；'quit' 退出。")
    while True:
        try:
            line = input("pose = ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        if line.strip().lower() in ("q","quit","exit"): break

        seed = None
        if "seed=" in line:
            parts = line.split("seed="); pose_str = parts[0].rstrip(", ")
            try:
                arr = np.array(ast.literal_eval(parts[1]), dtype=float).reshape(-1)
                if arr.size != 6: raise ValueError
                seed = np.deg2rad(arr)
            except Exception:
                print("seed 解析失败，忽略。"); seed = None
        else:
            pose_str = line

        try:
            T = parse_pose(pose_str)
        except Exception as e:
            print("位姿解析错误：", e); continue

        if solver == "analytic":
            q0 = seed if seed is not None else np.zeros(robot.n)
            try:
                q_best, sols = ik_ur5_analytic_wrt_robot(robot, T, q_previous=q0)
            except Exception as e:
                print("解析 IK 失败：", e); continue
            q_pick = _ik_solution_menu(robot, T, sols, q_seed=q0, ax=ax, lim=lim)
            print("最终选择的解（deg）=", np.round(np.rad2deg(q_pick), 2))
            plot_stick(robot, q_pick, ax=ax, lim=lim, show_frames=True)
        else:
            q0 = seed if seed is not None else np.zeros(robot.n)
            ok, q_num, info = ik_numeric(
                robot, T, q0=q0, seeds_extra=seed,
                solver="LM", tol=1e-6, ilimit=300, slimit=300,
                n_random=6, rand_scale_deg=20.0
            )
            if not ok or q_num is None:
                print(f"数值 IK 失败：{info}"); continue
            print("数值 IK 解（deg）=", np.round(np.rad2deg(q_num), 2))
            plot_stick(robot, q_num, ax=ax, lim=lim, show_frames=True)

    plt.ioff(); plt.show(); print("退出 IK。")

# ---------- CLI 入口 ----------
def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lim", type=float, default=0.8, help="可视化坐标范围")
    parser.add_argument("--mode", choices=["ask","fk","ik"], default="ask",
                        help="启动模式：ask=运行时询问、fk、ik（默认 ask）")
    parser.add_argument("--ik-solver", choices=["numeric","analytic"], default="numeric",
                        help="IK 求解器选择")
    args = parser.parse_args()

    robot = make_ur5()

    mode = args.mode
    if mode == "ask":
        ans = input("请选择模式（fk / ik）：").strip().lower()
        mode = "ik" if ans == "ik" else "fk"

    if mode == "fk":
        interactive_fk(robot, lim=args.lim)
    else:
        interactive_ik(robot, lim=args.lim, solver=args.ik_solver)
