#!/usr/bin/env python3
"""
Image Augmentation Script for Shirt Dataset
データセット拡張スクリプト：
- 回転、シフト、輝度変化、ズームなどの変換を適用
- 各クラスを100枚以上に拡張
- 幾何学的不変性を学習させるための多様性確保
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import random


def augment_image(image_path, output_dir, target_count=100):
    """
    1つの画像を複数のバージョンで拡張

    Args:
        image_path: 入力画像のパス
        output_dir: 出力先ディレクトリ
        target_count: 目標数（例：5個の拡張バージョンを生成）
    """
    img = Image.open(image_path)
    base_name = Path(image_path).stem

    augmentations = []

    # 1. 回転（Rotation）
    for angle in [-15, -10, -5, 5, 10, 15]:
        rotated = img.rotate(angle, expand=False, fillcolor="white")
        augmentations.append(("rotate", angle, rotated))

    # 2. 水平フリップ（Horizontal Flip）
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    augmentations.append(("flip", "h", flipped))

    # 3. 垂直フリップ（Vertical Flip）
    flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
    augmentations.append(("flip", "v", flipped_v))

    # 4. 輝度調整（Brightness）
    enhancer = ImageEnhance.Brightness(img)
    for factor in [0.7, 0.85, 1.15, 1.3]:
        brightened = enhancer.enhance(factor)
        augmentations.append(("brightness", factor, brightened))

    # 5. コントラスト調整（Contrast）
    enhancer = ImageEnhance.Contrast(img)
    for factor in [0.8, 1.2]:
        contrasted = enhancer.enhance(factor)
        augmentations.append(("contrast", factor, contrasted))

    # 6. ズーム（Zoom in）
    width, height = img.size
    for zoom_factor in [1.1, 1.2]:
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        zoomed = img.resize(new_size, Image.LANCZOS)
        # クロップして元のサイズに
        left = (new_size[0] - width) // 2
        top = (new_size[1] - height) // 2
        cropped = zoomed.crop((left, top, left + width, top + height))
        augmentations.append(("zoom", zoom_factor, cropped))

    # 7. シフト（Shift）
    for shift_x, shift_y in [(10, 5), (-10, 5), (5, -10), (-5, -10)]:
        shifted = Image.new("RGB", img.size, "white")
        shifted.paste(img, (shift_x, shift_y))
        augmentations.append(("shift", (shift_x, shift_y), shifted))

    # 8. 色合い調整（Color）
    enhancer = ImageEnhance.Color(img)
    for factor in [0.9, 1.1]:
        colored = enhancer.enhance(factor)
        augmentations.append(("color", factor, colored))

    # 原画像を含める
    augmentations.append(("original", 0, img))

    # 拡張画像を保存
    os.makedirs(output_dir, exist_ok=True)
    for idx, (aug_type, param, augmented_img) in enumerate(augmentations):
        output_path = os.path.join(output_dir, f"{base_name}_aug_{idx:02d}.jpg")
        augmented_img.save(output_path, "JPEG", quality=95)
        print(f"  Saved: {os.path.basename(output_path)}")

    return len(augmentations)


def expand_dataset(dataset_base_path, target_per_class=100):
    """
    全データセットを拡張

    Args:
        dataset_base_path: datasetフォルダのパス
        target_per_class: 各クラスの目標数
    """
    classes = ["ironed_white_shirt", "wrinkled_white_shirt"]

    for class_name in classes:
        class_dir = os.path.join(dataset_base_path, class_name)
        output_dir = os.path.join(dataset_base_path, f"{class_name}_augmented")

        if not os.path.exists(class_dir):
            print(f"❌ クラス {class_name} が見つかりません")
            continue

        # 既存ファイル数をカウント
        existing_files = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        print(f"\n📁 {class_name}")
        print(f"   現在の画像数: {len(existing_files)}枚")

        # 出力ディレクトリをクリア（再実行対応）
        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)

        total_augmented = 0
        for image_file in sorted(existing_files):
            image_path = os.path.join(class_dir, image_file)
            print(f"   処理中: {image_file}")
            count = augment_image(image_path, output_dir)
            total_augmented += count

        print(f"   ✅ 拡張完了: 全{total_augmented}枚を生成")
        print(f"   出力先: {output_dir}")


if __name__ == "__main__":
    dataset_path = "/Users/maitaguri/Documents/B3/Experiments/EdgeAI/dataset"

    print("=" * 60)
    print("🖼️  Image Augmentation for Shirt Dataset")
    print("=" * 60)

    expand_dataset(dataset_path, target_per_class=100)

    print("\n" + "=" * 60)
    print("✨ データセット拡張完了！")
    print("=" * 60)
