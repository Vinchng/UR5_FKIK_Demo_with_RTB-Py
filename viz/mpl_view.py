# viz/mpl_view.py
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3

def _fk_points(robot, q):
    Ts = [robot.base if isinstance(robot.base, SE3) else SE3(robot.base)]
    T  = Ts[0]
    for i, L in enumerate(robot.links):
        Ai = L.A(q[i]);  Ai = Ai if isinstance(Ai, SE3) else SE3(Ai)
        T = T * Ai
        Ts.append(T)
    pts = np.array([Ti.t for Ti in Ts])
    return Ts, pts

def _triad(ax, T, length=0.1, lw=2):
    p, R = T.t, T.R
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.quiver(p[0], p[1], p[2],
                  R[0,i]*length, R[1,i]*length, R[2,i]*length,
                  color=colors[i], linewidth=lw)

def plot_stick(robot, q, ax=None, lim=0.8, show_frames=True, trail=None):
    """
    画一帧骨架图（用于交互式 FK/IK 刷新）
    """
    Ts, pts = _fk_points(robot, q)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.cla()

    # 地面网格（简洁）
    gx = np.linspace(-lim, lim, 15)
    gy = np.linspace(-lim, lim, 15)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    ax.plot_wireframe(GX, GY, GZ, linewidth=0.35, alpha=0.28)

    # 连杆 + 关节
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', linewidth=3.0, markersize=6)

    # 末端/基座三轴
    if show_frames:
        _triad(ax, Ts[0], length=0.10, lw=2)
        _triad(ax, Ts[-1], length=0.10, lw=2.5)

    # 可选：末端轨迹
    if trail is not None and len(trail) > 1:
        ax.plot(trail[:,0], trail[:,1], trail[:,2], linewidth=2.0, alpha=0.8)

    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([0, 2*lim])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=25, azim=-60)
    plt.draw(); plt.pause(0.001)
