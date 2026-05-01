#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_compose.py - 將書法字畫合成到宋式室內照片中

使用方式: python 2_compose.py
功能:
  1. 從 Pexels 下載宋式室內參考圖
  2. 互動式選擇字畫 + 室內圖，點選牆面四個角落，自動合成輸出
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, requests, tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import cv2
from dotenv import load_dotenv

# ── 路徑設定 ──────────────────────────────────────────────
# 打包成 .exe 後，__file__ 指向暫存目錄；改用執行檔所在目錄
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

ARTWORKS_DIR = BASE_DIR / 'artworks'
NOBG_DIR     = BASE_DIR / 'artworks_nobg'
ROOMS_DIR    = BASE_DIR / 'rooms'
OUTPUT_DIR   = BASE_DIR / 'output'
# 優先使用去背版本
CATALOG_PATH = NOBG_DIR / 'catalog_nobg.json'
if not CATALOG_PATH.exists():
    CATALOG_PATH = ARTWORKS_DIR / 'catalog.json'
ENV_PATH     = BASE_DIR / '.env'

load_dotenv(ENV_PATH)
PEXELS_KEY = os.getenv('PEXELS_API_KEY', '')

# Pexels 搜尋詞（宋式/中式室內）
PEXELS_QUERIES = [
    "chinese interior design",
    "chinese study room",
    "zen interior design",
    "asian minimalist interior",
    "chinese calligraphy room",
    "new chinese style interior",
    "oriental living room",
]

HEADERS_PEXELS = {"Authorization": PEXELS_KEY}


# ── Pexels 下載 ───────────────────────────────────────────

def fetch_rooms():
    """從 Pexels 下載宋式室內參考圖"""
    if not PEXELS_KEY:
        print("ERROR: .env 中找不到 PEXELS_API_KEY")
        return

    ROOMS_DIR.mkdir(exist_ok=True)
    total = 0

    for query in PEXELS_QUERIES:
        print(f"\n搜尋: {query}")
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers=HEADERS_PEXELS,
                params={"query": query, "per_page": 5, "orientation": "landscape"},
                timeout=15
            )
            if resp.status_code != 200:
                print(f"  API 回應 {resp.status_code}")
                continue

            photos = resp.json().get("photos", [])
            for i, photo in enumerate(photos, 1):
                img_url = photo["src"]["large2x"]
                pid     = photo["id"]
                fname   = f"room_{pid}.jpg"
                fpath   = ROOMS_DIR / fname

                if fpath.exists():
                    print(f"  已存在: {fname}")
                    continue

                print(f"  [{i}/{len(photos)}] 下載 {fname} ...", end=" ", flush=True)
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    fpath.write_bytes(img_resp.content)
                    size_kb = len(img_resp.content) // 1024
                    print(f"OK ({size_kb} KB)")
                    total += 1
                else:
                    print("失敗")
        except Exception as e:
            print(f"  錯誤: {e}")

    print(f"\n下載完成，共 {total} 張新圖片")
    print(f"儲存位置: {ROOMS_DIR}")


# ── 圖像處理 ──────────────────────────────────────────────

def remove_white_bg(img_pil):
    """
    書法字畫去底色：自動偵測紙張底色（白/米白/淺灰均適用）
    - 比底色暗的像素 → 墨跡（保留，alpha=255）
    - 接近底色的像素 → 透明（alpha=0）
    - 紅色印章 → 強制保留（alpha=255）
    """
    rgb = np.array(img_pil.convert('RGB'))
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    h, w = gray.shape
    s = max(20, min(80, h // 12, w // 12))
    bg_samples = np.concatenate([
        gray[:s, :s].flatten(), gray[:s, -s:].flatten(),
        gray[-s:, :s].flatten(), gray[-s:, -s:].flatten(),
    ])
    bg = np.percentile(bg_samples, 92)

    # 比底色暗 25 以上才視為墨跡（平滑過渡）
    ink_thresh = bg - 25
    alpha = np.clip((bg - gray) / max(bg - ink_thresh, 1) * 255, 0, 255).astype(np.uint8)

    # 保留紅色印章：高紅、低綠低藍
    r, g_ch, b_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_seal = (r > 140) & (r > g_ch * 1.5) & (r > b_ch * 1.5)
    alpha[red_seal] = 255

    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, 'RGBA')


def normalize_to_white(img_pil):
    """
    將字畫底色標準化為純白（不論原本是白/米白/灰）
    以四角採樣偵測底色，線性拉伸讓底色 → 255，墨跡保持暗色
    """
    arr = np.array(img_pil.convert('RGB')).astype(np.float32)
    h, w = arr.shape[:2]
    s = max(20, min(80, h // 12, w // 12))
    corners = np.concatenate([
        arr[:s, :s].reshape(-1, 3),
        arr[:s, -s:].reshape(-1, 3),
        arr[-s:, :s].reshape(-1, 3),
        arr[-s:, -s:].reshape(-1, 3),
    ])
    bg = np.percentile(corners, 90, axis=0)          # 每個通道的底色亮度
    scale = 255.0 / np.maximum(bg, 30)               # 拉伸比例（避免除以零）
    normalized = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, 'RGB')


def build_mounted_artwork(artwork_pil, frame_ratio=0.07):
    """
    建立裱褙效果：米色宣紙框 + 深棕內線 + 紙張紋理
    回傳 (mounted_rgb, frame_mask, art_mask) — 皆為 numpy array
    """
    from PIL import ImageDraw
    aw, ah = artwork_pil.size
    short  = min(aw, ah)
    border = max(28, int(short * frame_ratio))   # 外框寬度
    inner  = max(4,  int(border * 0.13))         # 深色內線寬度

    tw = aw + 2 * (border + inner)
    th = ah + 2 * (border + inner)
    xo, yo = border + inner, border + inner       # 字畫起始座標

    # 米色宣紙底色（加細微隨機紋理）
    np.random.seed(7)
    base = np.full((th, tw, 3), [243, 235, 217], dtype=np.float32)
    base += np.random.normal(0, 4.5, (th, tw, 3))
    mounted = np.clip(base, 0, 255).astype(np.uint8)

    # 貼入已標準化的字畫
    mounted[yo:yo+ah, xo:xo+aw] = np.array(artwork_pil)

    # 畫深棕色內線（仿絲質裱條）
    mounted_pil = Image.fromarray(mounted)
    draw = ImageDraw.Draw(mounted_pil)
    for i in range(inner):
        draw.rectangle(
            [xo - inner + i, yo - inner + i,
             xo + aw + inner - i - 1, yo + ah + inner - i - 1],
            outline=(78, 50, 24)
        )
    mounted = np.array(mounted_pil)

    # 遮罩：框區域 / 字畫區域（互斥）
    frame_mask = np.ones((th, tw), dtype=np.uint8) * 255
    frame_mask[yo:yo+ah, xo:xo+aw] = 0

    art_mask = np.zeros((th, tw), dtype=np.uint8)
    art_mask[yo:yo+ah, xo:xo+aw] = 255

    return mounted, frame_mask, art_mask, (tw, th)


def composite_artwork(room_path, artwork_path, wall_pts):
    """
    透視合成主函式：原始字畫 + 裱褙框直接貼上牆面 + 陰影
    字畫完整清晰呈現，不做任何混色處理
    wall_pts：裱褙後整幅作品的四個角落（含框），順序：左上→右上→右下→左下
    """
    room    = Image.open(room_path).convert('RGB')
    artwork = Image.open(artwork_path).convert('RGB')   # 原始掃描，不做任何處理

    rw, rh = room.size

    # 放大室內圖至最小 2800px 寬，確保字畫細節清晰可讀
    TARGET_W = 2800
    if rw < TARGET_W:
        scale_up = TARGET_W / rw
        room = room.resize((int(rw * scale_up), int(rh * scale_up)), Image.LANCZOS)
        wall_pts = [(x * scale_up, y * scale_up) for x, y in wall_pts]
        rw, rh = room.size

    # 建立裱褙（字畫原色 + 米色外框 + 深棕內線）
    mounted, frame_mask, art_mask, (tw, th) = build_mounted_artwork(artwork)

    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = np.float32(wall_pts)
    M   = cv2.getPerspectiveTransform(src, dst)

    warp_kw = dict(flags=cv2.INTER_LANCZOS4,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    mask_kw = dict(flags=cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped_img = cv2.warpPerspective(mounted, M, (rw, rh), **warp_kw)

    # 整體遮罩（框 + 字畫）
    all_mask_src = np.clip(
        frame_mask.astype(np.int32) + art_mask.astype(np.int32), 0, 255
    ).astype(np.uint8)
    warped_all = cv2.warpPerspective(all_mask_src, M, (rw, rh), **mask_kw)
    inside = (warped_all > 128).astype(np.float32)
    in3    = np.stack([inside, inside, inside], axis=2)

    room_f = np.array(room).astype(np.float32)
    img_f  = warped_img.astype(np.float32)

    # 直接貼上：作品範圍內用字畫，範圍外保留室內圖
    result = room_f * (1 - in3) + img_f * in3

    # 陰影：整幅作品向右下偏移，高斯模糊，30% 不透明
    shift    = max(5, int(min(rw, rh) * 0.005))
    shadow   = cv2.GaussianBlur(warped_all, (0, 0), sigmaX=shift * 1.5)
    shadow_s = np.zeros_like(shadow)
    shadow_s[shift:, shift:] = shadow[:-shift, :-shift]
    shadow_s[warped_all > 64] = 0
    sm3 = np.stack([shadow_s, shadow_s, shadow_s], axis=2).astype(np.float32) / 255.0 * 0.35

    result = result * (1 - sm3)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), 'RGB')


# ── 互動式牆面選取 UI ─────────────────────────────────────

class CornerSelector:
    """
    顯示室內圖，讓使用者點選上方兩個角落（左上→右上），
    系統依字畫比例自動算出下方兩點，之後可拖拉全部 4 點微調。
    """

    CORNER_LABELS = ["左上角", "右上角", "右下角", "左下角"]
    CORNER_COLORS = ["#e74c3c", "#27ae60", "#2980b9", "#f39c12"]
    MAX_W, MAX_H  = 1000, 660
    DRAG_RADIUS   = 16

    def __init__(self, room_path, aspect_ratio=1.5):
        """
        aspect_ratio: 裱褙後字畫的 高/寬 比例，用於自動計算下方兩點
        """
        self.room_path    = room_path
        self.orig         = Image.open(room_path)
        self.aspect_ratio = aspect_ratio   # th / tw
        self.points       = []             # (x_orig, y_orig)，最多 4 點
        self.scale_x      = 1.0
        self.scale_y      = 1.0
        self._drag_idx    = None
        self._confirm_btn = None

    def run(self):
        root = tk.Tk()
        root.title("點選字畫位置 — 涵晞草堂字畫展示")
        root.resizable(False, False)

        orig_w, orig_h = self.orig.size
        scale = min(self.MAX_W / orig_w, self.MAX_H / orig_h, 1.0)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        self.scale_x = orig_w / disp_w
        self.scale_y = orig_h / disp_h

        display_img = self.orig.resize((disp_w, disp_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(display_img)

        tk.Label(root,
            text="步驟：① 點左上角  ② 點右上角  —  下方兩角自動依比例產生，可拖拉微調",
            font=("Microsoft JhengHei", 11), pady=6).pack()

        self.canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img, tags="bg")

        self.status = tk.Label(root,
            text="第 1 步：點選字畫左上角",
            font=("Microsoft JhengHei", 11), fg="#c0392b", pady=4)
        self.status.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="重新選取", width=10,
                  command=self._reset).pack(side='left', padx=8)
        tk.Button(btn_frame, text="取消", width=10,
                  command=root.destroy).pack(side='left', padx=8)

        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root = root
        root.mainloop()

        return self.points if len(self.points) == 4 else None

    # ── 座標轉換 ──────────────────────────────────────────
    def _to_disp(self, x_orig, y_orig):
        return x_orig / self.scale_x, y_orig / self.scale_y

    def _to_orig(self, x_disp, y_disp):
        return x_disp * self.scale_x, y_disp * self.scale_y

    def _nearest_corner(self, x_disp, y_disp):
        for i, (xo, yo) in enumerate(self.points):
            cx, cy = self._to_disp(xo, yo)
            if ((x_disp - cx) ** 2 + (y_disp - cy) ** 2) ** 0.5 <= self.DRAG_RADIUS:
                return i
        return None

    # ── 依比例自動補全下方兩點 ───────────────────────────
    def _auto_complete(self):
        """由 TL、TR 計算 BR、BL，維持字畫原始長寬比"""
        x1, y1 = self.points[0]   # TL（原始座標）
        x2, y2 = self.points[1]   # TR
        dx, dy = x2 - x1, y2 - y1
        w = (dx * dx + dy * dy) ** 0.5
        if w < 1:
            return
        h = w * self.aspect_ratio
        # 垂直於上邊、向下的單位向量（順時針旋轉 90°）
        px = -dy / w * h
        py =  dx / w * h
        # 若下方向量實際朝上（py < 0），翻轉
        if py < 0:
            px, py = -px, -py
        if len(self.points) == 2:
            self.points.append((x2 + px, y2 + py))   # BR
            self.points.append((x1 + px, y1 + py))   # BL
        else:
            self.points[2] = (x2 + px, y2 + py)
            self.points[3] = (x1 + px, y1 + py)

    # ── 重繪所有標記 ─────────────────────────────────────
    def _redraw(self):
        self.canvas.delete("markers")
        disp = [self._to_disp(x, y) for x, y in self.points]

        if len(disp) >= 2:
            for i in range(len(disp) - 1):
                self.canvas.create_line(*disp[i], *disp[i+1],
                    fill="white", width=2, dash=(8, 4), tags="markers")
        if len(disp) == 4:
            self.canvas.create_line(*disp[3], *disp[0],
                fill="white", width=2, dash=(8, 4), tags="markers")

        r = 10
        for i, (cx, cy) in enumerate(disp):
            col = self.CORNER_COLORS[i]
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                fill=col, outline="white", width=2, tags="markers")
            self.canvas.create_text(cx, cy, text=str(i+1),
                fill="white", font=("Arial", 9, "bold"), tags="markers")

    # ── 事件處理 ─────────────────────────────────────────
    def _on_press(self, event):
        x, y = event.x, event.y

        if len(self.points) == 4:
            idx = self._nearest_corner(x, y)
            if idx is not None:
                self._drag_idx = idx
                self.canvas.config(cursor="fleur")
            return

        self.points.append(self._to_orig(x, y))

        if len(self.points) == 1:
            self.status.config(text="第 2 步：點選字畫右上角")
            self._redraw()
        elif len(self.points) == 2:
            self._auto_complete()   # 自動補 BR、BL
            self._redraw()
            self._on_all_placed()

    def _on_drag(self, event):
        if self._drag_idx is None:
            return
        self.points[self._drag_idx] = self._to_orig(event.x, event.y)
        # 拖拉上方任一點時，同步更新下方兩點以維持比例
        if self._drag_idx in (0, 1):
            self._auto_complete()
        self._redraw()

    def _on_release(self, event):
        self._drag_idx = None
        if len(self.points) == 4:
            self.canvas.config(cursor="hand2")

    def _on_all_placed(self):
        self.status.config(text="拖拉上方圓點可整體調整比例；拖拉下方圓點可單獨微調。滿意後點「確認合成」")
        self.canvas.config(cursor="hand2")
        if self._confirm_btn is None:
            self._confirm_btn = tk.Button(
                self.root, text="確認合成", width=14,
                font=("Microsoft JhengHei", 11), bg="#27ae60", fg="white",
                command=self.root.destroy)
            self._confirm_btn.pack(pady=6)

    def _reset(self):
        self.points.clear()
        self._drag_idx = None
        self._confirm_btn = None
        self.canvas.delete("all")
        self.canvas.config(cursor="crosshair")
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img, tags="bg")
        self.status.config(text="第 1 步：點選字畫左上角")


# ── 合成主流程 UI ─────────────────────────────────────────

def list_rooms():
    rooms = sorted(ROOMS_DIR.glob("*.jpg")) + sorted(ROOMS_DIR.glob("*.png"))
    return rooms


def list_artworks():
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, encoding='utf-8') as f:
        catalog = json.load(f)
    return catalog


def show_image_preview(path, title="預覽"):
    """用 tkinter 顯示圖片預覽"""
    win = tk.Tk()
    win.title(title)
    img = Image.open(path)
    img.thumbnail((900, 650), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    tk.Label(win, image=tk_img).pack()
    tk.Button(win, text="關閉", command=win.destroy, width=12).pack(pady=6)
    win.mainloop()


def compose_flow():
    """引導使用者完成合成流程"""
    catalog = list_artworks()
    rooms   = list_rooms()

    if not catalog:
        print("找不到字畫目錄，請先執行 1_scraper.py")
        return

    # ── 選室內圖 ──────────────────────────────────────────
    print("\n【步驟 1：選擇室內圖】")
    if rooms:
        print("已下載的室內圖：")
        for i, r in enumerate(rooms, 1):
            print(f"  {i:2d}. {r.name}")
        print(f"   0. 從本機自行選擇圖片")
        choice = input("請輸入編號: ").strip()
        if choice == "0":
            root = tk.Tk(); root.withdraw()
            room_path = filedialog.askopenfilename(
                title="選擇室內圖",
                filetypes=[("圖片", "*.jpg *.jpeg *.png *.webp")])
            root.destroy()
            if not room_path:
                print("未選擇圖片，取消")
                return
            room_path = Path(room_path)
        else:
            try:
                room_path = rooms[int(choice) - 1]
            except (ValueError, IndexError):
                print("無效的選擇")
                return
    else:
        print("尚無室內圖，請先執行「下載室內圖」（選項 1）")
        print("或自行選擇圖片：")
        root = tk.Tk(); root.withdraw()
        room_path = filedialog.askopenfilename(
            title="選擇室內圖",
            filetypes=[("圖片", "*.jpg *.jpeg *.png *.webp")])
        root.destroy()
        if not room_path:
            return
        room_path = Path(room_path)

    print(f"已選室內圖: {room_path.name}")

    # ── 選字畫 ────────────────────────────────────────────
    print("\n【步驟 2：選擇字畫】")
    categories = list(catalog.keys())
    for i, cat in enumerate(categories, 1):
        count = len(catalog[cat])
        png   = sum(1 for x in catalog[cat] if x['type'].startswith('PNG'))
        print(f"  {i:2d}. {cat}  ({count} 張，PNG:{png})")

    cat_idx = input("選擇類別編號: ").strip()
    try:
        category = categories[int(cat_idx) - 1]
    except (ValueError, IndexError):
        print("無效的選擇")
        return

    items = catalog[category]
    print(f"\n{category} 的字畫：")
    for i, item in enumerate(items, 1):
        print(f"  {i:3d}. {item['filename']}  {item['type']}")

    art_idx = input("選擇字畫編號 (直接 Enter 選第 1 張): ").strip()
    art_idx = art_idx if art_idx else "1"
    try:
        art_item   = items[int(art_idx) - 1]
        art_path   = Path(art_item['file'])
    except (ValueError, IndexError):
        print("無效的選擇")
        return

    if not art_path.exists():
        print(f"找不到字畫檔案: {art_path}")
        return

    print(f"已選字畫: {art_item['filename']}")

    # ── 點選牆面位置 ──────────────────────────────────────
    print("\n【步驟 3：點選字畫位置】")
    print("即將開啟室內圖，請點選字畫左上角和右上角，下方兩點自動依比例產生")
    input("按 Enter 開啟圖片...")

    # 預先計算裱褙後的長寬比，傳給選取器
    _art_tmp = Image.open(art_path).convert('RGB')
    _, _, _, (_tw, _th) = build_mounted_artwork(_art_tmp)
    aspect = _th / _tw

    selector = CornerSelector(str(room_path), aspect_ratio=aspect)
    wall_pts  = selector.run()

    if not wall_pts or len(wall_pts) != 4:
        print("未完成角落選取，取消")
        return

    print(f"牆面座標: {[f'({int(x)},{int(y)})' for x,y in wall_pts]}")

    # ── 合成 ──────────────────────────────────────────────
    print("\n【步驟 4：合成中...】")
    OUTPUT_DIR.mkdir(exist_ok=True)

    out_name = f"result_{room_path.stem}_{art_item['filename']}"
    out_path  = OUTPUT_DIR / out_name.replace('.png', '.jpg').replace(' ', '_')
    # 確保副檔名為 jpg
    out_path  = out_path.with_suffix('.jpg')

    result = composite_artwork(str(room_path), str(art_path), wall_pts)
    result.save(str(out_path), "JPEG", quality=92)

    print(f"合成完成！儲存至: {out_path}")
    print("開啟預覽...")
    show_image_preview(str(out_path), title=f"展示效果 — {art_item['filename']}")


# ── 主選單 ────────────────────────────────────────────────

def main():
    ROOMS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("  涵晞草堂 字畫展示合成工具")
    print("=" * 50)

    while True:
        rooms_count   = len(list(ROOMS_DIR.glob("*.jpg")))
        catalog       = list_artworks()
        artworks_count = sum(len(v) for v in catalog.values())

        print(f"\n  室內圖: {rooms_count} 張 | 字畫素材: {artworks_count} 張")
        print()
        print("  1. 下載宋式室內參考圖（Pexels）")
        print("  2. 開始合成字畫到室內圖")
        print("  3. 開啟輸出資料夾")
        print("  0. 離開")
        print()

        choice = input("請選擇: ").strip()

        if choice == "1":
            fetch_rooms()
        elif choice == "2":
            compose_flow()
        elif choice == "3":
            os.startfile(str(OUTPUT_DIR))
        elif choice == "0":
            print("再見！")
            break
        else:
            print("請輸入 0–3")


if __name__ == "__main__":
    main()
