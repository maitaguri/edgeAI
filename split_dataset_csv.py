#!/usr/bin/env python3
"""
Split the unified dataset.csv into separate training and validation CSV files
統合されたdataset.csvを訓練用と検証用に分割
"""

import csv
import os


def split_dataset_csv(input_csv_path, dataset_base_path):
    """
    Split dataset.csv into separate train and validation CSV files

    Args:
        input_csv_path: 統合されたCSVファイルのパス
        dataset_base_path: datasetフォルダのパス
    """

    train_csv_path = os.path.join(dataset_base_path, "train_dataset.csv")
    validation_csv_path = os.path.join(dataset_base_path, "validation_dataset.csv")

    train_rows = []
    validation_rows = []

    print("=" * 73)
    print("📂 Splitting dataset.csv into train and validation files...")
    print("=" * 73)

    # Read the unified CSV file
    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                image_path, label, data_type = row[0], row[1], row[2]

                if data_type == "train":
                    train_rows.append([image_path, label])
                elif data_type == "validation":
                    validation_rows.append([image_path, label])

    # Write training CSV
    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(train_rows)

    print(f"\n✅ Training CSV作成完了！")
    print(f"   出力ファイル: {train_csv_path}")
    print(f"   合計行数: {len(train_rows) + 1}行（ヘッダー含む）")
    print(f"   データ行数: {len(train_rows)}行")

    # Write validation CSV
    with open(validation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(validation_rows)

    print(f"\n✅ Validation CSV作成完了！")
    print(f"   出力ファイル: {validation_csv_path}")
    print(f"   合計行数: {len(validation_rows) + 1}行（ヘッダー含む）")
    print(f"   データ行数: {len(validation_rows)}行")

    # Statistics
    train_label_0 = sum(1 for row in train_rows if row[1] == "0")
    train_label_1 = sum(1 for row in train_rows if row[1] == "1")
    val_label_0 = sum(1 for row in validation_rows if row[1] == "0")
    val_label_1 = sum(1 for row in validation_rows if row[1] == "1")

    print("\n" + "=" * 73)
    print("📊 分割統計:\n")
    print("   【Training データ】")
    print(
        f"   アイロン済み (label=0): {train_label_0}枚 ({train_label_0/len(train_rows)*100:.1f}%)"
    )
    print(
        f"   しわあり (label=1): {train_label_1}枚 ({train_label_1/len(train_rows)*100:.1f}%)"
    )
    print(f"   合計: {len(train_rows)}枚")

    print("\n   【Validation データ】")
    print(
        f"   アイロン済み (label=0): {val_label_0}枚 ({val_label_0/len(validation_rows)*100:.1f}%)"
    )
    print(
        f"   しわあり (label=1): {val_label_1}枚 ({val_label_1/len(validation_rows)*100:.1f}%)"
    )
    print(f"   合計: {len(validation_rows)}枚")

    print("\n" + "=" * 73)


if __name__ == "__main__":
    dataset_path = "/Users/maitaguri/Documents/B3/Experiments/EdgeAI/dataset"
    input_csv = os.path.join(dataset_path, "dataset.csv")

    split_dataset_csv(input_csv, dataset_path)
