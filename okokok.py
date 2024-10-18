import os
import glob
import pandas as pd
import argparse
import json
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

def save_top_k_mean_stats(root_dir, stats_to_compute):
    # 初始化每个标签对应的 step 和数值列表
    values_per_step = {stat['tag']: {} for stat in stats_to_compute}

    # 获取所有的 rank_number 文件夹
    rank_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for rank_folder in rank_folders:
        event_files = glob.glob(os.path.join(rank_folder, 'events.out.tfevents.*'))
        for event_file in event_files:
            tags = [stat['tag'] for stat in stats_to_compute]
            data = extract_scalars_from_events(event_file, tags)

            for stat in stats_to_compute:
                tag = stat['tag']
                # 提取数据
                for step, value in data.get(tag, []):
                    step = int (step / 100)
                    if step not in values_per_step[tag]:
                        values_per_step[tag][step] = []
                    values_per_step[tag][step].append(value)

    # 获取所有的 steps
    all_steps = set()
    for stat in stats_to_compute:
        all_steps.update(values_per_step[stat['tag']].keys())
    all_steps = sorted(all_steps)

    results = []

    for step in all_steps:
        result_row = {'step': step}
        for stat in stats_to_compute:
            tag = stat['tag']
            column_name = stat['column_name']
            k = stat.get('k', None)
            sort_order = stat.get('sort_order', 'desc')
            agg_func = stat.get('agg_func', 'mean')

            values = values_per_step[tag].get(step, [])
            if values:
                reverse = (sort_order == 'desc')
                values_sorted = sorted(values, reverse=reverse)
                k = min(k, len(values_sorted)) if k is not None else None
                if k is not None:
                    top_k_values = values_sorted[:k]
                else:
                    top_k_values = values_sorted
                # 应用聚合函数
                if agg_func == 'mean':
                    aggregated_value = sum(top_k_values) / len(top_k_values)
                elif agg_func == 'median':
                    n = len(top_k_values)
                    mid = n // 2
                    if n % 2 == 0:
                        median = (top_k_values[mid - 1] + top_k_values[mid]) / 2
                    else:
                        median = top_k_values[mid]
                    aggregated_value = median
                else:
                    # 支持其他聚合函数
                    try:
                        aggregated_value = getattr(pd.Series(top_k_values), agg_func)()
                    except AttributeError:
                        print(f"Unsupported aggregation function: {agg_func}")
                        aggregated_value = None
            else:
                aggregated_value = None

            result_row[column_name] = aggregated_value

        results.append(result_row)

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)
    # 按照 step 排序
    df.sort_values(by='step', inplace=True)
    # 保存到 CSV 文件
    output_csv = os.path.join(root_dir, os.path.basename(root_dir) + '_stats.csv')
    df.to_csv(str(output_csv), index=False)
    print(f"结果已保存到 {output_csv}")

if __name__ == '__main__':

    target_path = "data/1016"

    stats_to_compute = [
        {
        "tag": "Top-1 Accuracy/test",
        "column_name": "top_1_accuracy",
        "k": None,
        "sort_order": "desc",
        "agg_func": "mean"
        },
        {
            "tag": "Top-3 Accuracy/test",
            "column_name": "top_3_accuracy",
            "k": None,
            "sort_order": "desc",
            "agg_func": "mean"
        },
    ]

    for sub_dir_path in os.listdir(target_path):
        sub_dir_full_path = os.path.join(target_path, sub_dir_path)
        if os.path.isdir(sub_dir_full_path):
            save_top_k_mean_stats(sub_dir_full_path, stats_to_compute)