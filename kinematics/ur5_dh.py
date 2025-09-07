import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3   # 别忘了导入 SE3

def make_ur5():
    """
    Build UR5 via standard D-H parameters (meters / radians).
    """
    d1, a2, a3, d4, d5, d6 = 0.08916, -0.425, -0.39225, 0.10915, 0.09465, 0.0823
    links = [
        RevoluteDH(a=0.0,     alpha=np.pi/2, d=d1),
        RevoluteDH(a=a2,      alpha=0.0,     d=0.0),
        RevoluteDH(a=a3,      alpha=0.0,     d=0.0),
        RevoluteDH(a=0.0,     alpha=np.pi/2, d=d4),
        RevoluteDH(a=0.0,     alpha=-np.pi/2,d=d5),
        RevoluteDH(a=0.0,     alpha=0.0,     d=d6),
    ]

    ur5 = DHRobot(links, name='UR5_DH')

    # 加上你求出来的 tool 外参
    ur5.tool = SE3([[-1.0000000e+00, -6.1232340e-17,  1.2246468e-16, -3.0915000e-01],
                    [ 1.2246468e-16,  6.1232340e-17,  1.0000000e+00,  8.0490000e-02],
                    [-6.1232340e-17,  1.0000000e+00, -6.1232340e-17, -7.7455000e-01],
                    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

    return ur5
