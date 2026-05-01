#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_remove_bg.py
批次去背：將所有字畫去除底色，輸出為透明 PNG
結果存在 artworks_nobg/ 資料夾（原始圖不動）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json, time
import numpy as np
from pathlib import Path
from PIL import Image

ARTWORKS_DIR = Path(__file__).parent / 'artworks'
NOBG_DIR     = Path(__file__).parent / 'artworks_nobg'
CATALOG_IN   = ARTWORKS_DIR / 'catalog.json'
CATALOG_OUT  = NOBG_DIR / 'catalog_nobg.json'


def remove_bg(img_pil):
    """
    自動偵測紙張底色並去除：
    - 取四角採樣估計背景亮度
    - 比背景暗的像素 → 墨跡（保留，alpha 漸變）
    - 接近底色的像素 → 透明
    - 紅色印章 → 強制不透明保留
    """
    rgb  = np.array(img_pil.convert('RGB'))
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    h, w = gray.shape
    s = max(20, min(80, h // 12, w // 12))
    corners = np.concatenate([
        gray[:s, :s].flatten(),
        gray[:s, -s:].flatten(),
        gray[-s:, :s].flatten(),
        gray[-s:, -s:].flatten(),
    ])
    bg = np.percentile(corners, 92)

    # 比背景暗 25 開始出現墨跡，平滑過渡
    ink_thresh = bg - 25
    alpha = np.clip(
        (bg - gray) / max(bg - ink_thresh, 1) * 255, 0, 255
    ).astype(np.uint8)

    # 紅色印章強制不透明
    r, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = (r > 140) & (r > g_ch * 1.5) & (r > b_ch * 1.5)
    alpha[red] = 255

    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3]  = alpha
    return Image.fromarray(rgba, 'RGBA')


def format_size(b):
    if b >= 1024 * 1024:
        return f"{b/1024/1024:.1f} MB"
    return f"{b/1024:.0f} KB"


def main():
    if not CATALOG_IN.exists():
        print("找不到 artworks/catalog.json，請先執行 1_scraper.py")
        return

    with open(CATALOG_IN, encoding='utf-8') as f:
        catalog = json.load(f)

    NOBG_DIR.mkdir(exist_ok=True)

    total_items = sum(len(v) for v in catalog.values())
    done = 0
    skipped = 0
    failed = 0
    catalog_out = {}

    print("=" * 52)
    print("  批次去背工具 — 涵晞草堂字畫")
    print(f"  共 {total_items} 張，輸出至 artworks_nobg/")
    print("=" * 52)

    for category, items in catalog.items():
        cat_dir = NOBG_DIR / category
        cat_dir.mkdir(exist_ok=True)
        catalog_out[category] = []

        print(f"\n【{category}】共 {len(items)} 張")

        for item in items:
            src = Path(item['file'])
            if not src.exists():
                print(f"  找不到原始檔: {src.name}")
                failed += 1
                continue

            out_name = src.stem + '_nobg.png'
            out_path = cat_dir / out_name

            new_entry = {
                **item,
                'file': str(out_path),
                'filename': out_name,
                'type': 'PNG（去背）',
            }

            if out_path.exists() and out_path.stat().st_size > 1000:
                size = out_path.stat().st_size
                print(f"  已存在: {out_name} ({format_size(size)})")
                skipped += 1
                catalog_out[category].append(new_entry)
                continue

            try:
                img = Image.open(src)
                result = remove_bg(img)
                result.save(str(out_path), 'PNG', optimize=False)
                size = out_path.stat().st_size
                done += 1
                print(f"  [{done+skipped:3d}/{total_items}] {out_name} ({format_size(size)})")
                catalog_out[category].append(new_entry)
            except Exception as e:
                print(f"  ERROR {src.name}: {e}")
                failed += 1

    # 儲存新目錄
    with open(CATALOG_OUT, 'w', encoding='utf-8') as f:
        json.dump(catalog_out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 52)
    print(f"完成:  {done} 張")
    print(f"已存在: {skipped} 張（跳過）")
    print(f"失敗:  {failed} 張")
    print(f"\n去背圖片位置: {NOBG_DIR}")
    print(f"新目錄檔案:   {CATALOG_OUT}")
    print("\n合成時請改用 artworks_nobg/catalog_nobg.json")


if __name__ == '__main__':
    main()
