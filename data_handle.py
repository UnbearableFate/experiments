import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_scalars_from_events(event_file, tags):
    """
    从事件文件中提取指定标签的标量数据。

    参数：
        event_file (str): 事件文件路径。
        tags (list of str): 要提取的标签列表。

    返回：
        data (dict): 键为标签，值为 (step, value) 的列表。
    """
    event_acc = event_accumulator.EventAccumulator(event_file)
    event_acc.Reload()

    available_tags = event_acc.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        if tag in available_tags:
            scalars = event_acc.Scalars(tag)
            data[tag] = [(scalar.step, scalar.value) for scalar in scalars]
        else:
            print(f"'{tag}' not found in {event_file}")
            data[tag] = []
    return data


root_dir = './202410031713'  # 请替换为您的实际根目录路径

# 用于存储每个 step 对应的精度和时间列表
accuracies_per_step = {}  # key: step, value: list of accuracies
times_per_step = {}  # key: step, value: list of times

# 要提取的标签列表
tags_to_extract = ['Accuracy/test', 'Time/train']

# 获取所有的 rank_number 文件夹
rank_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for rank_folder in rank_folders:
    event_files = glob.glob(os.path.join(rank_folder, 'events.out.tfevents.*'))
    for event_file in event_files:
        data = extract_scalars_from_events(event_file, tags_to_extract)

        # 提取 Accuracy/test 数据
        for step, value in data['Accuracy/test']:
            if step not in accuracies_per_step:
                accuracies_per_step[step] = []
            accuracies_per_step[step].append(value)

        # 提取 Time/train 数据
        for step, value in data['Time/train']:
            if step not in times_per_step:
                times_per_step[step] = []
            times_per_step[step].append(value)

results = []

# 获取所有的 steps，并确保精度和时间数据都存在
all_steps = sorted(set(accuracies_per_step.keys()) & set(times_per_step.keys()))

for step in all_steps:
    # 处理精度数据
    accuracies = accuracies_per_step[step]
    accuracies_sorted = sorted(accuracies, reverse=True)
    top4_accuracies = accuracies_sorted[:4]
    if len(top4_accuracies) > 0:
        mean_top4_accuracy = sum(top4_accuracies) / len(top4_accuracies)
    else:
        mean_top4_accuracy = None

    # 处理时间数据
    times = times_per_step[step]
    times_sorted = sorted(times)
    top4_times = times_sorted[:8]
    if len(top4_times) > 0:
        mean_top4_time = sum(top4_times) / len(top4_times)
    else:
        mean_top4_time = None

    # 将结果添加到列表
    results.append({
        'step': step,
        'mean_top4_accuracy': mean_top4_accuracy,
        'mean_top4_time': mean_top4_time
    })

# 将结果转换为 DataFrame
df = pd.DataFrame(results)
# 按照 step 排序
df.sort_values(by='step', inplace=True)
# 保存到 CSV 文件
output_csv = 'async_kfac_no_delay.csv'
df.to_csv(output_csv, index=False)

print(f"Mean top 4 accuracies and times per step have been saved to {output_csv}")