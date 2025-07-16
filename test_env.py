import robosuite
# from robosuite.controllers import load_composite_controller_config
from robosuite import load_controller_config
from env.robosuite_wrapper import PixelRobosuite
# BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
controller_config = load_controller_config(default_controller="OSC_POSE")

# create an environment to visualize on-screen
env = robosuite.make(
    "NutAssemblySquare",
    robots=["Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=False,                      # on-screen rendering
    render_camera="agentview",              # visualize the "frontview" camera
    has_offscreen_renderer=True,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=300,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    reward_shaping = False
)
env2 = PixelRobosuite(env_name="NutAssemblySquare",
                      robots=["Panda"],
                      episode_length=300)
def main():
    obs = env2.reset()

if __name__ == "__main__":
    main()