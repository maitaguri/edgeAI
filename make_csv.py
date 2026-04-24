import pandas as pd
import os

# 設定
dataset_path = "dataset"
classes = ["ironed_white_shirt", "wrinkled_white_shirt"]
output_file = "index.csv"

data = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # NNCが読み込める形式（相対パス, ラベル）
            file_path = os.path.join(dataset_path, class_name, filename)
            data.append([file_path, label])

# CSVとして保存
df = pd.DataFrame(data, columns=["x:image", "y:label"])
df = df.sample(frac=1).reset_index(drop=True)  # シャッフル（学習に重要！）
df.to_csv(output_file, index=False)

print(f"成功！ {output_file} が作成されました。画像数: {len(df)}")
