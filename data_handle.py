import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_accuracy_from_events(event_file, tag='Accuracy/test'):
    event_acc = event_accumulator.EventAccumulator(event_file)
    event_acc.Reload()
    
    tags = event_acc.Tags()
    if 'scalars' in tags and tag in tags['scalars']:
        scalars = event_acc.Scalars(tag)
        return [(scalar.step, scalar.value) for scalar in scalars]
    else:
        print(f"'{tag}' not found in {event_file}")
        return []

root_dir = '/work/NBB/yu_mingzhe/experiments/202410031648'  # 请替换为您的实际根目录路径

accuracies_per_step = {}  # key: step, value: list of accuracies

rank_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for rank_folder in rank_folders:
    event_files = glob.glob(os.path.join(rank_folder, 'events.out.tfevents.*'))
    for event_file in event_files:
        accuracies = extract_accuracy_from_events(event_file)
        for step, value in accuracies:
            if step not in accuracies_per_step:
                accuracies_per_step[step] = []
            accuracies_per_step[step].append(value)

results = []

for step in sorted(accuracies_per_step.keys()):
    accuracies = accuracies_per_step[step]
    accuracies_sorted = sorted(accuracies, reverse=True)
    top4 = accuracies_sorted[:4]
    mean_top4_accuracy = sum(top4) / len(top4)
    results.append({'step': step, 'accuracy': mean_top4_accuracy})

df = pd.DataFrame(results)
df.sort_values(by='step', inplace=True)
output_csv = 'mean_top4_accuracies.csv'
df.to_csv(output_csv, index=False)

print(f"Mean top 4 accuracies per step have been saved to {output_csv}")