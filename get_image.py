from icrawler.builtin import BingImageCrawler  # GoogleからBingに変更
import os

# 保存先フォルダを作成
os.makedirs("dataset/wrinkled_white_shirt", exist_ok=True)
os.makedirs("dataset/ironed_white_shirt", exist_ok=True)

# 1. シワあり白シャツの収集 (Bingを使用)
crawler = BingImageCrawler(storage={"root_dir": "dataset/wrinkled_white_shirt"})
crawler.crawl(keyword="wrinkled white shirt photo texture", max_num=50)

# 2. アイロン済み白シャツの収集 (Bingを使用)
crawler = BingImageCrawler(storage={"root_dir": "dataset/ironed_white_shirt"})
crawler.crawl(keyword="ironed white shirt formal photo", max_num=50)

print("ダウンロードが完了しました！datasetフォルダを確認してください。")
