from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import pandas as pd
import math
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=int, default=1, help='1:first floor, 2: second floor')
    parser.add_argument('--task', type=int, default=1, help='1:task1, 2: task2')
    args = parser.parse_args()
    return args
args = parse_args()
if args.task == 1:
    if not os.path.isdir('./data_task1'):
        os.mkdir('./data_task1')
else:
    if args.floor == 1:
        if not os.path.isdir('./data_task2_floor1'):
            os.mkdir('./data_task2_floor1')
    else:
        if not os.path.isdir('./data_task2_floor2'):
            os.mkdir('./data_task2_floor2')
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    if args.task == 1:
        # camera from Top-down view for task1
        rgb_sensor_spec1 = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec1.uuid = "color_sensor_top"
        rgb_sensor_spec1.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec1.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec1.position = [0.0, 2.6, 0.0]
        rgb_sensor_spec1.orientation = [
            -math.pi/2,
            0.0,
            0.0,
        ]
        rgb_sensor_spec1.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    if args.task == 1:
        agent_cfg.sensor_specifications = [rgb_sensor_spec, rgb_sensor_spec1, depth_sensor_spec, semantic_sensor_spec]
    else:
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
if args.floor == 1:
    agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
else:
    agent_state.position = np.array([0.0, 1.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")

i = 0
def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        if args.task == 1:
            cv2.imshow("RGB_top", transform_rgb_bgr(observations["color_sensor_top"]))
        else:
            cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
            cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")       
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        if args.task == 1:
            if args.floor == 1:
                cv2.imwrite(f'./data_task1/front_view1.png', transform_rgb_bgr(observations["color_sensor"]))
                cv2.imwrite(f'./data_task1/top_view1.png', transform_rgb_bgr(observations["color_sensor_top"]))
            else:
                cv2.imwrite(f'./data_task1/front_view2.png', transform_rgb_bgr(observations["color_sensor"]))
                cv2.imwrite(f'./data_task1/top_view2.png', transform_rgb_bgr(observations["color_sensor_top"]))
        else:
            global i
            # save picture
            if args.floor == 1:
                cv2.imwrite(f'./data_task2_floor1/{i}_color.png', transform_rgb_bgr(observations["color_sensor"]))
                cv2.imwrite(f'./data_task2_floor1/{i}_depth.png', transform_depth(observations["depth_sensor"]))
                cv2.imwrite(f'./data_task2_floor1/{i}_semantic.png', transform_semantic(observations["semantic_sensor"]))
            else:
                cv2.imwrite(f'./data_task2_floor2/{i}_color.png', transform_rgb_bgr(observations["color_sensor"]))
                cv2.imwrite(f'./data_task2_floor2/{i}_depth.png', transform_depth(observations["depth_sensor"]))
                cv2.imwrite(f'./data_task2_floor2/{i}_semantic.png', transform_semantic(observations["semantic_sensor"]))
            i += 1
        return sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z

x_list, y_list, z_list, rw_list, rx_list, ry_list, rz_list = [], [], [], [], [], [], []
action = "move_forward"
x, y, z, rw, rx, ry, rz = navigateAndSee(action)
x_list.append(x)
y_list.append(y)
z_list.append(z)
rw_list.append(rw)
rx_list.append(rx)
ry_list.append(ry)
rz_list.append(rz)


while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        x, y, z, rw, rx, ry, rz = navigateAndSee(action)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        rw_list.append(rw)
        rx_list.append(rx)
        ry_list.append(ry)
        rz_list.append(rz)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        x, y, z, rw, rx, ry, rz = navigateAndSee(action)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        rw_list.append(rw)
        rx_list.append(rx)
        ry_list.append(ry)
        rz_list.append(rz)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        x, y, z, rw, rx, ry, rz = navigateAndSee(action)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        rw_list.append(rw)
        rx_list.append(rx)
        ry_list.append(ry)
        rz_list.append(rz)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        if args.task == 1:
            break
        else:
            if args.floor == 1:
                output = {'x':x_list,'y':y_list,'z':z_list, 'rw':rw_list, 'rx':rx_list, 'ry':ry_list, 'rz':rz_list}
                output_df = pd.DataFrame(output)
                output_df.to_csv("./data_task2_floor1/camera_pose.csv", index=False)
            else:
                output = {'x':x_list,'y':y_list,'z':z_list, 'rw':rw_list, 'rx':rx_list, 'ry':ry_list, 'rz':rz_list}
                output_df = pd.DataFrame(output)
                output_df.to_csv("./data_task2_floor2/camera_pose.csv", index=False)
            break
    else:
        print("INVALID KEY")
        continue
