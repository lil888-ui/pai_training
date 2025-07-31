#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import argparse
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

# CRANE+ の関節名 (InferenceNode と同じ)
ARM_JOINT_NAMES    = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class DatasetInferenceNode(Node):
    def __init__(self, dataset: LeRobotDataset, episode_index: int, rate: float, policy_path: str, device: str):
        super().__init__('crane_plus_dataset_infer')

        # publishers
        self.joint_pub = self.create_publisher(JointState,      '/joint_states', 10)
        self.arm_pub   = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.grip_pub  = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)

        # load policy
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.eval()
        self.device = torch.device(device)
        self.policy.to(self.device)

        # prepare episode frame indices
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx   = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = list(range(from_idx, to_idx))

        self.dataset = dataset
        self.rate    = rate
        self.idx     = 0

        # timer at dataset rate
        self.timer = self.create_timer(1.0/self.rate, self.step)

        self.get_logger().info(f"DatasetInferenceNode: playing ep#{episode_index} at {rate}Hz")

    def step(self):
        # finish?
        if self.idx >= len(self.frame_ids):
            self.get_logger().info("Episode finished.")
            rclpy.shutdown()
            return

        # fetch sample
        sample = self.dataset[self.frame_ids[self.idx]]
        state = sample["observation.state"]           # shape (5,)
        env   = sample["observation.environment_state"]
        # publish joint_states (so Gazebo or robot shows the "known" state)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        js.position = state.tolist()
        self.joint_pub.publish(js)

        # prepare obs for inference
        obs = {
            "observation.state":             torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
            "observation.environment_state": torch.tensor(env,   dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()  # (5,)

        # publish arm trajectory
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = action[:4].tolist()
        pt.time_from_start.sec     = 0
        pt.time_from_start.nanosec = int(1.0/self.rate * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        # publish gripper trajectory
        jt_g = JointTrajectory()
        jt_g.joint_names = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(action[4])]
        pt_g.time_from_start.sec     = 0
        pt_g.time_from_start.nanosec = int(1.0/self.rate * 1e9)
        jt_g.points.append(pt_g)
        self.grip_pub.publish(jt_g)

        self.idx += 1

def main():
    parser = argparse.ArgumentParser(description="Replay known dataset + inference on CRANE+")
    parser.add_argument('--dataset-root',  type=str, required=True, help='path to dataset root')
    parser.add_argument('--repo-id',       type=str, required=True, help='dataset repo_id')
    parser.add_argument('--episode-index', type=int, default=0, help='which episode')
    parser.add_argument('--policy-path',   type=str, required=True, help='path to trained policy directory')
    parser.add_argument('--device',        type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--rate',          type=float, default=50.0, help='playback & inference rate (Hz)')
    args = parser.parse_args()

    # load dataset
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)

    rclpy.init()
    node = DatasetInferenceNode(dataset, args.episode_index, args.rate, args.policy_path, args.device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
