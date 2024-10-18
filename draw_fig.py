import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 目录路径和统计项的定义
directory_path = "data/1016"
title = 'ResNet18 CIFAR10'

# 需要统计的内容（可以包含多个）
statistics = ["top_1_accuracy","top_3_accuracy"]  # 添加更多需要统计的项

# スムージングのスパン（適宜調整）
smoothing_span = 8

# 使用 pathlib 遍历所有子目录中的 CSV 文件
csv_files = list(Path(directory_path).rglob("*.csv"))

# 为每个 statistic 单独创建图表并保存
for statistic in statistics:
    plt.figure(figsize=(10, 6))  # 创建新的图表

    # 遍历所有找到的 CSV 文件
    subfix = ""
    for file_path in csv_files:
        # 读取 CSV 文件
        data = pd.read_csv(file_path)

        # 检查是否包含要统计的列
        if statistic not in data.columns:
            print(f"Warning: '{statistic}' not found in {file_path}. Skipping this file.")
            continue

        # 指数移動平均でデータをスムージング
        data['smoothed_value'] = data[statistic].ewm(span=smoothing_span).mean()

        # スムーズな曲線を描画
        data['step'] /= 10  # 将时间从百毫秒转换为秒
        plt.plot(data['step'], data['smoothed_value'], label=file_path.stem)

    # グラフのタイトルと軸ラベルを設定
    plt.title(f"{title} - {statistic}")
    plt.xlabel('time/s')
    plt.ylabel(statistic)

    # 凡例を表示
    plt.legend()

    # 保存文件名，格式为：directory_path/statistic_resnet18_cifar10.pdf
    save_path = os.path.join(directory_path, f"{statistic}_resnet18_cifar10.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()  # 关闭图表，释放内存
    print(f"Saved {statistic} graph to {save_path}")