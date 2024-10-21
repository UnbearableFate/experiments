import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def compute_time_differences(root_path, method1, method2, target_accuracies):
    """
    Calculate the time differences between two training methods when reaching target accuracies.

    Parameters:
    - root_path: str, the path to the root data directory.
    - method1: str, the name of the first method.
    - method2: str, the name of the second method.
    - target_accuracies: list of float, the list of target accuracies.

    Functionality:
    - Traverse CSV files under the root directory that match the naming rules and extract the time required to reach target accuracies.
    - For each delay value, calculate the time difference between the two methods for each target accuracy.
    - Output a CSV file containing delays and time differences.
    - Plot a graph where each target accuracy corresponds to a line showing the time difference versus delay.
    """
    method_data = {method1: {}, method2: {}}
    delay_set = set()

    # Find all CSV files matching the pattern
    pattern = os.path.join(root_path, '*_delay_*/', '*_stats.csv')
    csv_files = glob.glob(pattern)

    for csv_file in csv_files:
        # Extract method name and delay from directory name
        dir_name = os.path.basename(os.path.dirname(csv_file))
        # Example of dir_name: 'allreduce_delay_0000'

        if '_delay_' in dir_name:
            method, delay_str = dir_name.split('_delay_')
            # Convert delay string to integer (milliseconds)
            delay_ms = int(delay_str)

            if method not in [method1, method2]:
                continue  # Skip methods we're not interested in

            delay_set.add(delay_ms)

            # Read CSV file
            df = pd.read_csv(csv_file)

            for target_accuracy in target_accuracies:
                # Check if 'top_1_accuracy' column exists
                if 'top_1_accuracy' not in df.columns:
                    print(f"Missing 'top_1_accuracy' column in file: {csv_file}")
                    continue

                # Find indices where top_1_accuracy >= target_accuracy
                indices = df.index[df['top_1_accuracy'] >= target_accuracy].tolist()

                if len(indices) >= 3:
                    # Get the index of the third occurrence
                    idx = indices[2]
                    # Get the corresponding step (training time in ms)
                    time_ms = df.loc[idx, 'step']

                    # Store the time
                    if delay_ms not in method_data[method]:
                        method_data[method][delay_ms] = {}
                    method_data[method][delay_ms][target_accuracy] = time_ms
                else:
                    # Did not reach target accuracy three times, skip or handle accordingly
                    pass

    # Calculate time differences
    delays = sorted(list(delay_set))
    output_data = []

    for delay in delays:
        row = {'Delay (ms)': delay}
        for target_accuracy in target_accuracies:
            time1 = method_data[method1].get(delay, {}).get(target_accuracy, None)
            time2 = method_data[method2].get(delay, {}).get(target_accuracy, None)

            if time1 is not None and time2 is not None:
                time_diff = time1 - time2
                row[f"Time Difference for Target Accuracy {target_accuracy}"] = time_diff
            else:
                row[f"Time Difference for Target Accuracy {target_accuracy}"] = None  # or use np.nan
        output_data.append(row)

    # Convert to DataFrame
    output_df = pd.DataFrame(output_data)

    # Save to CSV file
    output_csv_path = os.path.join(root_path, 'time_differences.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Time differences saved to {output_csv_path}")

    # Plot the graph
    plt.figure()
    for target_accuracy in target_accuracies:
        y_values = output_df[f"Time Difference for Target Accuracy {target_accuracy}"] / 10  # Convert to seconds
        plt.plot(output_df['Delay (ms)'], y_values, marker='o', label=f"Target Accuracy {target_accuracy}")
    plt.xlabel('Delay (ms)')
    plt.ylabel('Time Difference (s)')
    plt.title('Time Difference vs. Delay')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(root_path, 'time_differences_plot.png'))
    plt.show()

def compute_time_differences_with_zero_delay(root_path, method, target_accuracies):
    """
    Calculate, for the same method, the difference in training time to reach target accuracies between different delays and the zero-delay case.

    Parameters:
    - root_path: str, the path to the root data directory.
    - method: str, the name of the method.
    - target_accuracies: list of float, the list of target accuracies.

    Functionality:
    - Traverse CSV files under the root directory that match the naming rules and extract the time required to reach target accuracies.
    - For each delay value, calculate the time difference compared to the zero-delay case for each target accuracy.
    - Output a CSV file containing delays and time differences.
    - Plot a graph where each target accuracy corresponds to a line showing the time difference versus delay.
    """
    method_data = {}
    delay_set = set()

    # Find all CSV files matching the pattern
    pattern = os.path.join(root_path, f'{method}_delay_*/', f'{method}_delay_*_stats.csv')
    csv_files = glob.glob(pattern)

    for csv_file in csv_files:
        # Extract delay from directory name
        dir_name = os.path.basename(os.path.dirname(csv_file))
        # Example of dir_name: 'method_delay_0000'

        if '_delay_' in dir_name:
            _, delay_str = dir_name.split('_delay_')
            # Convert delay string to integer (milliseconds)
            delay_ms = int(delay_str)

            delay_set.add(delay_ms)

            # Read CSV file
            df = pd.read_csv(csv_file)

            for target_accuracy in target_accuracies:
                # Check if 'top_1_accuracy' column exists
                if 'top_1_accuracy' not in df.columns:
                    print(f"Missing 'top_1_accuracy' column in file: {csv_file}")
                    continue

                # Find indices where top_1_accuracy >= target_accuracy
                indices = df.index[df['top_1_accuracy'] >= target_accuracy].tolist()

                if len(indices) >= 3:
                    # Get the index of the third occurrence
                    idx = indices[2]
                    # Get the corresponding step (training time in ms)
                    time_ms = df.loc[idx, 'step']

                    # Store the time
                    if delay_ms not in method_data:
                        method_data[delay_ms] = {}
                    method_data[delay_ms][target_accuracy] = time_ms
                else:
                    # Did not reach target accuracy three times, skip or handle accordingly
                    pass

    # Get the zero-delay times
    zero_delay_times = method_data.get(0, {})

    # Calculate time differences compared to zero delay
    delays = sorted([d for d in delay_set if d != 0])
    output_data = []

    for delay in delays:
        row = {'Delay (ms)': delay}
        for target_accuracy in target_accuracies:
            time_zero = zero_delay_times.get(target_accuracy, None)
            time_delay = method_data.get(delay, {}).get(target_accuracy, None)

            if time_zero is not None and time_delay is not None:
                time_diff = time_delay - time_zero
                row[f"Time Difference for Target Accuracy {target_accuracy}"] = time_diff
            else:
                row[f"Time Difference for Target Accuracy {target_accuracy}"] = None  # or use np.nan
        output_data.append(row)

    # Convert to DataFrame
    output_df = pd.DataFrame(output_data)

    # Save to CSV file
    output_csv_path = os.path.join(root_path, f'{method}_time_differences_with_zero_delay.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Time differences with zero delay saved to {output_csv_path}")

    # Plot the graph
    plt.figure()
    for target_accuracy in target_accuracies:
        y_values = output_df[f"Time Difference for Target Accuracy {target_accuracy}"] #/ 10  # Convert to seconds
        plt.plot(output_df['Delay (ms)'], y_values, marker='o', label=f"Target Accuracy {target_accuracy}")
    plt.xlabel('Delay (ms)')
    plt.ylabel('Time Difference (s)')
    plt.title(f'Time Difference for {method} Compared to Zero Delay')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(root_path, f'{method}_time_differences_with_zero_delay_plot.png'))
    plt.show()

if __name__ == '__main__':
    root_path = "data/1016"
    method1 = "allreduce"
    method2 = "async"
    target_accuracies = [0.885, 0.89, 0.895]
    compute_time_differences(root_path, method1, method2, target_accuracies)
    #compute_time_differences_with_zero_delay(root_path, method2, target_accuracies)