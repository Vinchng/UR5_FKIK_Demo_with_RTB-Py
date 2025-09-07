from roboticstoolbox.backends import swift as rtb_swift

def launch_and_add(robot):
    """
    Launch Swift backend and add the robot.
    """
    backend = rtb_swift.Swift()
    backend.launch()
    backend.add(robot)
    return backend

def show_configuration(backend, robot, q):
    """
    Update robot configuration in Swift.
    """
    robot.q = q
    backend.step()
