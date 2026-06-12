import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    r"C:\Users\oohon\OneDrive\Desktop\Research\ifdl\RL\Mujoco\PPO_6DOF_Random\RL_PPO_6DOF\my_robot.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()