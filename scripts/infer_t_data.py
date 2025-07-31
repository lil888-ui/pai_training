#!/usr/bin/env python3
import argparse
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

# ジョイント名定義
ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class ActionPlaybackNode(Node):
    def __init__(self, dataset_root, repo_id, policy_path, episode_index, rate_hz=50):
        super().__init__('action_playback')
        # データセット読み込み
        self.dataset = LeRobotDataset(repo_id, root=dataset_root)
        from_idx = self.dataset.episode_data_index['from'][episode_index].item()
        to_idx   = self.dataset.episode_data_index['to'][episode_index].item()
        self.frame_ids = list(range(from_idx, to_idx))
        self.current = 0

        # ポリシー読み込み
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        # Publishers
        self.arm_pub     = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)

        # 再生レート
        self.dt    = 1.0 / rate_hz
        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        if self.current >= len(self.frame_ids):
            self.get_logger().info('Action playback finished.')
            rclpy.shutdown()
            return

        idx    = self.frame_ids[self.current]
        sample = self.dataset[idx]

        # 観測データをテンソル化
        obs = {
            'observation.state':             torch.tensor(sample['observation.state'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'observation.environment_state': torch.tensor(sample['observation.environment_state'], dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()  # shape (5,)

        # 時刻取得
        now = self.get_clock().now().to_msg()
        t   = self.dt * (self.current + 1)
        sec, nsec = int(t), int((t - int(t)) * 1e9)

        # アームアクションをパブリッシュ
        jt = JointTrajectory()
        jt.header.stamp = now
        jt.joint_names  = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions          = action[:4].tolist()
        pt.time_from_start.sec    = sec
        pt.time_from_start.nanosec= nsec
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        # グリッパーアクションをパブリッシュ
        jt_g = JointTrajectory()
        jt_g.header.stamp = now
        jt_g.joint_names  = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions          = [float(action[4])]
        pt_g.time_from_start.sec    = sec
        pt_g.time_from_start.nanosec= nsec
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)

        self.current += 1


def main():
    parser = argparse.ArgumentParser(description='Play back predicted actions in Gazebo')
    parser.add_argument('--dataset-root', type=str, required=True, help='データセットルートパス')
    parser.add_argument('--repo-id',      type=str, required=True, help='LeRobotDataset の repo_id')
    parser.add_argument('--policy-path',  type=str, required=True, help='学習済みポリシーのディレクトリパス')
    parser.add_argument('--episode-index',type=int, default=0, help='評価対象のエピソード番号')
    parser.add_argument('--rate',         type=float, default=50.0, help='再生レート (Hz)')
    args = parser.parse_args()

    # Gazebo で CRANE+ の launch を先に起動しておくこと
    rclpy.init()
    node = ActionPlaybackNode(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        policy_path=args.policy_path,
        episode_index=args.episode_index,
        rate_hz=args.rate
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
