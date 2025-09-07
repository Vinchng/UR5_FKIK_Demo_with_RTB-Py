# **UR5 FK/IK Demo with RTB-Py**

本项目基于 [Robotics Toolbox for Python (RTB-Py)](https://github.com/petercorke/robotics-toolbox-python)，实现了 **UR5 机械臂的正运动学 (FK)** 与 **逆运动学 (IK)** 演示，并提供 **数值解**与**解析解**两种 IK 方案。

通过 `matplotlib` 渲染，可以在 3D 窗口中交互式输入关节角 / 末端位姿，观察 UR5 的姿态。

---

## **已知问题与修复**

在新版 `numpy>=2.0` 环境下，RTB-Py 中 `roboticstoolbox/mobile/DistanceTransformPlanner.py` 使用了已被移除的 `numpy.disp`，会导致导入失败：

ImportError: cannot import name 'disp' from 'numpy'

### 解决方案  

手动修改该文件，替换前几行：

原始代码
from numpy import disp

修改为：

```python
try:
    from numpy import disp
except Exception:
    def disp(msg):
        print(msg)
```

保存后重新运行，即可避免错误。

## **环境安装**

pip install roboticstoolbox-python spatialmath-python matplotlib
Python 版本建议：>=3.9。

## **使用方法**

**运行程序**
`python main.py`
启动后会询问进入 FK 模式或 IK 模式。

**FK 模式**

输入关节角（度）：
`q = 0 -90 90 0 90 0`

输入弧度：
`q = rad: 0 -1.57 1.57 0 1.57 0`

**IK 模式**

默认使用 数值 IK (ik_numeric)。

输入末端位姿示例：

pose = T=[[0,-1,0,0.3],[1,0,0,0.2],[0,0,1,0.5],[0,0,0,1]]

或位置 + 欧拉角：

pose = p=[0.3,0.2,0.5], rpy=[0,90,0]

**切换到解析 IK**

运行时指定参数：

python main.py --mode ik --ik-solver analytic

解析 IK 会给出 8 组候选解，并进入交互菜单，可通过：

输入 0~7 选择解

n / p 翻页

b 回到最接近 seed 的解

q 退出菜单

## **项目结构**

```
├── main.py              # 启动入口（几行代码）
├── app.py               # 核心逻辑（FK/IK交互、菜单、解析工具）
├── kinematics/
│   ├── ur5_dh.py        # 定义 UR5 的 D-H 参数 + tool 外参
│   ├── ik_numeric.py    # 数值解 IK
│   └── ik_analytic_ur5.py  # 解析解 IK + 封装
└── viz/
    └── mpl_view.py      # matplotlib 渲染器
```



## **小技巧**

robot.tool 已根据标定结果设置，保证 FK/IK 与真实 TCP 对齐。

数值 IK 更稳健，解析 IK 在特殊姿态下可能切到等价分支，可通过 seed 参数引导。

## **鸣谢**

Peter Corke 的 Robotics Toolbox for Python

Rasmus Skovgaard Andersen 关于 UR5 解析 IK 的推导 https://blog.csdn.net/fengyu19930920/article/details/81144042

IK解析解参考项目：https://gitee.com/borunte-robot/IK_Solver_UR5
