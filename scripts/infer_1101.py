#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import argparse
import numpy as np
import csv
from datetime import datetime
import os

csv_path = "inference_log.csv"

from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('crane_plus_act_infer')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        self.last_action = np.zeros(5, dtype=np.float32)
        self.timer = self.create_timer(0.05, self.publish_action)  # 20Hz
        self.latest_joint_state = None
        self.write_header = not os.path.exists(csv_path)

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        gripper_pos = name_to_pos.get(GRIPPER_JOINT_NAME, 0.0)
        self.latest_joint_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)

    def publish_action(self):
        if self.latest_joint_state is None:
            return

        obs = {
            'observation.state': torch.tensor(
                self.latest_joint_state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0),  # (1, 5)
            'observation.environment_state': torch.tensor(
                self.latest_joint_state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0),  # (1, 5)
        }
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()  # (5,)
        self.last_action = action

        # Publish to arm
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = action[:4].tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        # Publish to gripper
        jt_g = JointTrajectory()
        jt_g.joint_names = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(action[4])]
        pt_g.time_from_start.sec = 0
        pt_g.time_from_start.nanosec = int(0.1 * 1e9)
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)

        print("action:", action[:4], "gripper:", action[4])

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if self.write_header:
                writer.writerow([
                    "timestamp", "action1", "action2", "action3", "action4", "gripper"
                ])
                self.write_header = False
            writer.writerow([datetime.now().isoformat()] + action.tolist())


def main():
    parser = argparse.ArgumentParser(
        description='ACTPolicy inference node for CRANE+ V2 (ROS2)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained ACTPolicy model directory'
    )
    args = parser.parse_args()
    rclpy.init()
    node = InferenceNode(args.model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()