import argparse
import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--repo-id', type=str, required=True, help='Dataset repo_id')
    parser.add_argument('--policy-path', type=str, required=True, help='Path to trained policy directory')
    parser.add_argument('--episode-index', type=int, default=0, help='Episode index to evaluate')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # データセット読み込み
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f"Loaded dataset: {len(dataset)} samples, {dataset.num_episodes} episodes")

    # サンプラーで1エピソード分だけ抽出
    sampler = EpisodeSampler(dataset, args.episode_index)

    # ポリシー読み込み
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # 評価
    errors = []
    for i in sampler:
        sample = dataset[i]
        obs = {
            "observation.state": torch.tensor(sample["observation.state"], dtype=torch.float32).unsqueeze(0).to(args.device),
            "observation.environment_state": torch.tensor(sample["observation.environment_state"], dtype=torch.float32).unsqueeze(0).to(args.device),
        }
        gt_action = torch.tensor(sample["action"], dtype=torch.float32).to(args.device)

        with torch.no_grad():
            pred_action = policy.select_action(obs).squeeze(0)

        print(f"\033[32m pred_action: {pred_action}\033[0m")
        print(f"\033[31m gt_action: {gt_action}\033[0m")

        error = torch.norm(pred_action - gt_action).item()
        errors.append(error)

    print(f"\nMean L2 error for episode {args.episode_index} over {len(errors)} frames: {sum(errors)/len(errors):.4f}")

if __name__ == "__main__":
    main()
