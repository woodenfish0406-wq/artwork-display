#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compress_for_web.py
將字畫壓縮為適合網頁的小尺寸版本（最長邊 1800px，JPEG 85%）
輸出到 artworks_web/ 資料夾，並更新 catalog_web.json
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
from pathlib import Path
from PIL import Image

BASE_DIR    = Path(__file__).parent
SRC_DIR     = BASE_DIR / 'artworks_nobg'
DST_DIR     = BASE_DIR / 'artworks_web'
SRC_CAT     = SRC_DIR / 'catalog_nobg.json'
ALT_CAT     = BASE_DIR / 'artworks' / 'catalog.json'
DST_CAT     = BASE_DIR / 'catalog_web.json'

MAX_PX      = 1800   # 最長邊上限
JPEG_Q      = 85


def compress_image(src_path, dst_path):
    img = Image.open(src_path)

    # RGBA → RGB (white background)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')

    # 等比縮小
    w, h = img.size
    if max(w, h) > MAX_PX:
        scale = MAX_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img.save(str(dst_path), 'JPEG', quality=JPEG_Q, optimize=True)
    return dst_path.stat().st_size


def format_size(b):
    if b >= 1024 * 1024:
        return f"{b/1024/1024:.1f} MB"
    return f"{b/1024:.0f} KB"


def main():
    # 選擇來源 catalog
    if SRC_CAT.exists():
        with open(SRC_CAT, encoding='utf-8') as f:
            catalog = json.load(f)
        use_nobg = True
    elif ALT_CAT.exists():
        with open(ALT_CAT, encoding='utf-8') as f:
            catalog = json.load(f)
        use_nobg = False
    else:
        print("找不到 catalog.json，請先執行 1_scraper.py")
        return

    DST_DIR.mkdir(exist_ok=True)
    catalog_web = {}
    total_src = 0
    total_dst = 0
    done = skipped = failed = 0
    total = sum(len(v) for v in catalog.values())

    print("=" * 52)
    print("  壓縮字畫供網頁使用")
    print(f"  共 {total} 張 → artworks_web/")
    print("=" * 52)

    for category, items in catalog.items():
        cat_dir = DST_DIR / category
        cat_dir.mkdir(exist_ok=True)
        catalog_web[category] = []

        print(f"\n【{category}】")

        for item in items:
            src = Path(item['file'])
            if not src.exists():
                print(f"  找不到: {src.name}")
                failed += 1
                continue

            dst_name = src.stem.replace('_nobg', '') + '.jpg'
            dst = cat_dir / dst_name

            web_entry = {
                'filename': dst_name,
                'category': category,
                'file':     str(dst),
                'url':      item.get('url', ''),
                'type':     'JPG（網頁壓縮版）',
            }

            if dst.exists() and dst.stat().st_size > 500:
                size = dst.stat().st_size
                total_dst += size
                skipped += 1
                catalog_web[category].append(web_entry)
                print(f"  已存在: {dst_name} ({format_size(size)})")
                continue

            src_size = src.stat().st_size
            total_src += src_size
            try:
                dst_size = compress_image(src, dst)
                total_dst += dst_size
                ratio = (1 - dst_size / max(src_size, 1)) * 100
                done += 1
                print(f"  [{done+skipped:3d}/{total}] {dst_name}  "
                      f"{format_size(src_size)} → {format_size(dst_size)} (-{ratio:.0f}%)")
                catalog_web[category].append(web_entry)
            except Exception as e:
                print(f"  ERROR {src.name}: {e}")
                failed += 1

    with open(DST_CAT, 'w', encoding='utf-8') as f:
        json.dump(catalog_web, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 52)
    print(f"完成:  {done} 張")
    print(f"跳過:  {skipped} 張（已存在）")
    print(f"失敗:  {failed} 張")
    if total_src > 0:
        print(f"原始大小: {format_size(total_src)}")
    print(f"壓縮後:   {format_size(total_dst)}")
    print(f"\n輸出資料夾: {DST_DIR}")
    print(f"Catalog:    {DST_CAT}")


if __name__ == '__main__':
    main()
