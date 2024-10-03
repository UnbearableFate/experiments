import matplotlib.pyplot as plt
import pandas as pd
import os

# CSVファイルが保存されているディレクトリのパス

directory_path = "./time_result"
title = 'ResNet18 CIFAR10'

statistic = "time"

# スムージングのスパン（適宜調整）
smoothing_span = 8

# 各CSVファイルを順に処理
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        csv_file = filename
    else:
        continue
    file_path = os.path.join(directory_path, csv_file)
    data = pd.read_csv(file_path)

    # 指数移動平均でデータをスムージング
    data['smoothed_value'] = data[statistic].ewm(span=smoothing_span).mean()

    # スムーズな曲線を描画
    plt.plot(data['step'], data['smoothed_value'], label=os.path.splitext(csv_file)[0])

# グラフのタイトルと軸ラベルを設定
plt.xlabel('epoch')
plt.ylabel(statistic)

# 凡例を表示
plt.legend()

plt.savefig(directory_path+f"/resnet18_cifar10.pdf", format='pdf')
# グラフを表示
plt.show()
