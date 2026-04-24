#!/usr/bin/env python3
"""
Create CSV file for NNC (Neural Network Console) Dataset Import
NeuralNetworkConsoleのデータセットインポート用CSVを生成
元画像と拡張画像の両方をまとめたデータセット
"""

import os
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_dataset_csv(dataset_base_path, output_csv_path):
    """
    元画像と拡張画像をまとめてCSVファイルを生成

    Args:
        dataset_base_path: datasetフォルダのパス
        output_csv_path: 出力するCSVファイルのパス
    """

    # ラベル定義
    label_map = {
        "ironed_white_shirt": 0,
        "ironed_white_shirt_augmented": 0,
        "wrinkled_white_shirt": 1,
        "wrinkled_white_shirt_augmented": 1,
    }

    image_paths = []
    labels = []

    print("=" * 73)
    print("🗂️  Collecting images for dataset...")
    print("=" * 73)

    for class_name, label in label_map.items():
        class_dir = os.path.join(dataset_base_path, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  {class_name} not found, skipping...")
            continue

        # クラスディレクトリ内のすべての画像ファイルを取得
        image_files = sorted(
            [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )

        print(f"\n📁 {class_name}")
        print(f"   画像数: {len(image_files)}枚")

        for image_file in image_files:
            # 相対パスでCSVに保存（NNCが読み込みやすくするため）
            image_path = os.path.join(class_name, image_file)
            image_paths.append(image_path)
            labels.append(label)

        label_name = "アイロン済み" if label == 0 else "しわあり"
        print(f"   ラベル: {label} ({label_name})")

    # Perform stratified train/validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Create data rows with train/validation type
    data_rows = []
    for path, label in zip(train_paths, train_labels):
        data_rows.append([path, label, "train"])
    for path, label in zip(val_paths, val_labels):
        data_rows.append([path, label, "validation"])

    # CSVファイルに出力
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow(["image_path", "label", "type"])
        # データ行
        writer.writerows(data_rows)

    print("\n" + "=" * 73)
    print(f"✅ CSV生成完了！")
    print(f"   出力ファイル: {output_csv_path}")
    print(f"   合計行数: {len(data_rows) + 1}行（ヘッダー含む）")
    print("=" * 73)

    # 全体の統計情報
    total_ironed = sum(1 for label in labels if label == 0)
    total_wrinkled = sum(1 for label in labels if label == 1)

    # Train データの統計
    train_ironed = sum(1 for label in train_labels if label == 0)
    train_wrinkled = sum(1 for label in train_labels if label == 1)

    # Validation データの統計
    val_ironed = sum(1 for label in val_labels if label == 0)
    val_wrinkled = sum(1 for label in val_labels if label == 1)

    print("\n📊 データセット統計:\n")
    print("   【全データ】")
    print(f"   アイロン済み (label=0): {total_ironed}枚")
    print(f"   しわあり (label=1): {total_wrinkled}枚")
    print(f"   合計: {len(labels)}枚")

    print("\n   【Training データ】")
    train_total = len(train_labels)
    train_ironed_pct = (train_ironed / train_total * 100) if train_total > 0 else 0
    train_wrinkled_pct = (train_wrinkled / train_total * 100) if train_total > 0 else 0
    train_pct_of_total = (train_total / len(labels) * 100) if len(labels) > 0 else 0
    print(f"   アイロン済み (label=0): {train_ironed}枚 ({train_ironed_pct:.1f}%)")
    print(f"   しわあり (label=1): {train_wrinkled}枚 ({train_wrinkled_pct:.1f}%)")
    print(f"   合計: {train_total}枚 ({train_pct_of_total:.1f}%)")

    print("\n   【Validation データ】")
    val_total = len(val_labels)
    val_ironed_pct = (val_ironed / val_total * 100) if val_total > 0 else 0
    val_wrinkled_pct = (val_wrinkled / val_total * 100) if val_total > 0 else 0
    val_pct_of_total = (val_total / len(labels) * 100) if len(labels) > 0 else 0
    print(f"   アイロン済み (label=0): {val_ironed}枚 ({val_ironed_pct:.1f}%)")
    print(f"   しわあり (label=1): {val_wrinkled}枚 ({val_wrinkled_pct:.1f}%)")
    print(f"   合計: {val_total}枚 ({val_pct_of_total:.1f}%)")
    print("\n" + "=" * 73)


if __name__ == "__main__":
    dataset_path = "/Users/maitaguri/Documents/B3/Experiments/EdgeAI/dataset"
    output_csv = os.path.join(dataset_path, "dataset.csv")

    create_dataset_csv(dataset_path, output_csv)
