import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv("../inference_log.csv", parse_dates=["timestamp"])

# グラフ描画の設定
plt.figure(figsize=(12, 6))

# 関節動作を折れ線で描画
for i in range(1, 5):  # action1〜action4
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    plt.plot(df["timestamp"].to_numpy(), df[f"action{i}"].to_numpy(), label=f"Joint {i}")

# グリッパー動作も追加（破線で）
plt.plot(df["timestamp"].to_numpy(), df["gripper"].to_numpy(), label="Gripper", linestyle="--", color="gray")

# グラフ装飾
plt.title("Inference Joint Trajectory Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Joint Position")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 表示
plt.show()
