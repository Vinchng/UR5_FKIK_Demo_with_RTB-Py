import numpy as np

def fkine(robot, q, deg=False):
    """
    Forward kinematics wrapper.
    q : iterable[6] joint angles
    deg : if True, interpret q as degrees
    return: spatialmath.SE3
    """
    q = np.asarray(q, dtype=float)
    if deg:
        q = np.deg2rad(q)
    if robot.n != len(q):
        raise ValueError(f"Expected {robot.n} joints, got {len(q)}")
    return robot.fkine(q)
