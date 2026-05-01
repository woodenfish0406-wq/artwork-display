#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_scraper.py
從 mingliangchiang.com 爬取所有書法字畫圖片，依類別分資料夾儲存
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
import re
import os
import time
import json
from pathlib import Path

BASE_URL = "https://www.mingliangchiang.com"
ARTWORK_DIR = Path(__file__).parent / "artworks"

# 所有書法頁面，依字畫類型分類
PAGES = {
    "條幅": [
        "/selected-work-2",   # 書法 > 條幅
        "/selected-work-5",   # 新書發表 > 條幅之一
        "/selected-work-11",  # 新書發表 > 條幅之二
        "/selected-work-7",   # 新書發表 > 條幅之三
    ],
    "橫幅": [
        "/press-3",           # 書法 > 橫幅
        "/selected-work-10",  # 新書發表 > 橫幅
    ],
    "對聯": [
        "/press-6",           # 書法 > 對聯
        "/selected-work-9",   # 新書發表 > 對聯
    ],
    "斗方小品": [
        "/press-2",           # 書法 > 小品
        "/selected-work-6",   # 新書發表 > 斗方小品
    ],
    "條屏": [
        "/selected-work-8",   # 新書發表 > 條屏
    ],
    "對開條幅": [
        "/selected-work-4",   # 書法 > 對開條幅
    ],
    "群書治要": [
        "/bio",               # 書法 > 群書治要
    ],
    "二十四詩品": [
        "/press",             # 書法 > 二十四詩品
    ],
    "藝概": [
        "/selected-work-3",   # 書法 > 藝概
    ],
    "其他": [
        "/selected-work-12",  # 新書發表 > 其他
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.mingliangchiang.com/",
}

# 要排除的系統圖片（logo、icon 等非字畫圖片）
EXCLUDE_PATTERNS = [
    "facebook", "twitter", "linkedin", "instagram",
    "logo", "icon", "favicon", "arrow", "button",
    "wix_mp_",  # Wix 系統媒體
]


def clean_wix_url(url):
    """
    移除 Wix 圖片尺寸參數，取得原始全解析度圖片
    有參數: https://static.wixstatic.com/media/HASH~mv2.jpg/v1/fill/w_940,...
    無參數: https://static.wixstatic.com/media/HASH~mv2.jpg
    """
    match = re.match(r'(https://static\.wixstatic\.com/media/[^/?#]+~mv2\.[a-zA-Z]+)', url)
    if match:
        return match.group(1)
    # 若沒有 ~mv2，嘗試擷取基本 URL
    match = re.match(r'(https://static\.wixstatic\.com/media/[^/?#]+\.[a-zA-Z]{3,4})', url)
    if match:
        return match.group(1)
    return url


def is_artwork_image(url):
    """判斷是否為字畫圖片（排除系統圖示）"""
    url_lower = url.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in url_lower:
            return False
    # 圖片檔案大小通常比較大，透過副檔名篩選
    if not url_lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        return False
    # 必須包含 ~mv2 特徵（Wix 媒體上傳的識別標記）
    if '~mv2' not in url_lower:
        return False
    return True


def extract_image_urls(html_text):
    """從 HTML 原始碼（含內嵌 JSON）提取所有 wixstatic.com 圖片 URL"""
    # 找所有 wixstatic.com 圖片 URL（包含 JSON 字串、data-src、src 等各種位置）
    pattern = r'https://static\.wixstatic\.com/media/[a-zA-Z0-9_%~\-\.]+\.[a-zA-Z]{3,4}'
    raw_urls = re.findall(pattern, html_text)

    cleaned = {}
    for url in raw_urls:
        # URL 可能被 JSON 編碼（斜線變成 \/)
        url = url.replace('\\/', '/')
        clean = clean_wix_url(url)
        if is_artwork_image(clean):
            # 用 hash 部分當 key 做去重
            key = re.search(r'([a-f0-9]{32})', clean)
            if key:
                cleaned[key.group(1)] = clean

    return list(cleaned.values())


def download_image(url, save_path):
    """下載單張圖片，回傳檔案大小（bytes）"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if resp.status_code == 200:
            content = resp.content
            with open(save_path, 'wb') as f:
                f.write(content)
            return len(content)
        else:
            print(f"    HTTP {resp.status_code}")
    except Exception as e:
        print(f"    錯誤: {e}")
    return 0


def scrape_page(path):
    """爬取一個頁面，回傳圖片 URL 列表"""
    url = BASE_URL + path
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            urls = extract_image_urls(resp.text)
            return urls
        else:
            print(f"    頁面 HTTP {resp.status_code}: {url}")
    except Exception as e:
        print(f"    爬取失敗 {url}: {e}")
    return []


def format_size(bytes_count):
    if bytes_count >= 1024 * 1024:
        return f"{bytes_count / 1024 / 1024:.1f} MB"
    return f"{bytes_count / 1024:.0f} KB"


def main():
    ARTWORK_DIR.mkdir(parents=True, exist_ok=True)

    catalog = {}
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    total_bytes = 0

    print("=" * 50)
    print("涵晞草堂 書法字畫圖片下載工具")
    print("=" * 50)

    for category, page_paths in PAGES.items():
        print(f"\n【{category}】")
        cat_dir = ARTWORK_DIR / category
        cat_dir.mkdir(exist_ok=True)

        # 爬取此類別的所有頁面
        all_urls = {}
        for path in page_paths:
            print(f"  掃描: {BASE_URL + path}")
            urls = scrape_page(path)
            for url in urls:
                key = re.search(r'([a-f0-9]{32})', url)
                if key:
                    all_urls[key.group(1)] = url
            time.sleep(1.5)

        if not all_urls:
            print(f"  未找到圖片")
            continue

        print(f"  找到 {len(all_urls)} 張圖片，開始下載...")

        catalog[category] = []
        url_list = list(all_urls.values())

        for i, img_url in enumerate(url_list, 1):
            ext = img_url.rsplit('.', 1)[-1].lower()
            if ext not in ('jpg', 'jpeg', 'png', 'webp'):
                ext = 'jpg'
            filename = f"{category}_{i:03d}.{ext}"
            save_path = cat_dir / filename

            entry = {
                "file": str(save_path),
                "filename": filename,
                "category": category,
                "url": img_url,
                "type": "PNG（掃描稿）" if ext == 'png' else "JPG（照片）"
            }

            if save_path.exists() and save_path.stat().st_size > 1000:
                size = save_path.stat().st_size
                print(f"  [{i:3d}/{len(url_list)}] 已存在: {filename} ({format_size(size)})")
                total_skipped += 1
                catalog[category].append(entry)
                continue

            print(f"  [{i:3d}/{len(url_list)}] 下載: {filename} ...", end=" ", flush=True)
            size = download_image(img_url, save_path)

            if size > 0:
                print(f"OK ({format_size(size)})")
                total_downloaded += 1
                total_bytes += size
                catalog[category].append(entry)
            else:
                print("失敗")
                total_failed += 1

            time.sleep(0.8)

    # 儲存圖片目錄 catalog.json
    catalog_path = ARTWORK_DIR / "catalog.json"
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # 輸出統計
    print("\n" + "=" * 50)
    print("下載完成統計")
    print("=" * 50)
    print(f"新下載:  {total_downloaded} 張（{format_size(total_bytes)}）")
    print(f"已存在:  {total_skipped} 張")
    print(f"失敗:    {total_failed} 張")
    print(f"\n圖片目錄: {catalog_path}")
    print(f"字畫資料夾: {ARTWORK_DIR}")

    # 各類別統計
    print("\n各類別圖片數量：")
    for cat, items in catalog.items():
        png_count = sum(1 for x in items if x['type'].startswith('PNG'))
        jpg_count = len(items) - png_count
        print(f"  {cat}: {len(items)} 張（PNG:{png_count} / JPG:{jpg_count}）")


if __name__ == "__main__":
    main()
