#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — 涵晞草堂字畫展示工具（網頁版）
執行方式: streamlit run app.py
"""

import io as _io
import json
from pathlib import Path

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import numpy as np
import cv2

# ── 路徑設定 ──────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
WEB_DIR      = BASE_DIR / 'artworks_web'
ROOMS_DIR    = BASE_DIR / 'rooms'
CATALOG_PATH = BASE_DIR / 'catalog_web.json'


# ── 圖像處理 ──────────────────────────────────────────────

def build_mounted_artwork(artwork_pil, frame_ratio=0.07):
    aw, ah = artwork_pil.size
    short  = min(aw, ah)
    border = max(28, int(short * frame_ratio))
    inner  = max(4,  int(border * 0.13))
    tw = aw + 2 * (border + inner)
    th = ah + 2 * (border + inner)
    xo, yo = border + inner, border + inner

    np.random.seed(7)
    base = np.full((th, tw, 3), [243, 235, 217], dtype=np.float32)
    base += np.random.normal(0, 4.5, (th, tw, 3))
    mounted = np.clip(base, 0, 255).astype(np.uint8)
    mounted[yo:yo+ah, xo:xo+aw] = np.array(artwork_pil)

    mounted_pil = Image.fromarray(mounted)
    draw = ImageDraw.Draw(mounted_pil)
    for i in range(inner):
        draw.rectangle(
            [xo-inner+i, yo-inner+i, xo+aw+inner-i-1, yo+ah+inner-i-1],
            outline=(78, 50, 24))
    mounted = np.array(mounted_pil)

    frame_mask = np.ones((th, tw), dtype=np.uint8) * 255
    frame_mask[yo:yo+ah, xo:xo+aw] = 0
    art_mask = np.zeros((th, tw), dtype=np.uint8)
    art_mask[yo:yo+ah, xo:xo+aw] = 255
    return mounted, frame_mask, art_mask, (tw, th)


def composite_artwork(room_pil, artwork_pil, wall_pts, blend=0.45):
    """
    blend: 光影融入強度 0.0（直接貼上）→ 1.0（完全跟隨牆面光影）
    """
    room = room_pil.copy()
    rw, rh = room.size

    TARGET_W = 2400
    if rw < TARGET_W:
        s = TARGET_W / rw
        room = room.resize((int(rw * s), int(rh * s)), Image.LANCZOS)
        wall_pts = [(x * s, y * s) for x, y in wall_pts]
        rw, rh = room.size

    mounted, frame_mask, art_mask, (tw, th) = build_mounted_artwork(artwork_pil)

    src = np.float32([[0,0],[tw,0],[tw,th],[0,th]])
    dst = np.float32(wall_pts)
    M   = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(mounted, M, (rw, rh),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))

    all_mask = np.clip(
        frame_mask.astype(np.int32) + art_mask.astype(np.int32), 0, 255
    ).astype(np.uint8)
    warped_mask = cv2.warpPerspective(all_mask, M, (rw, rh),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    inside = (warped_mask > 128).astype(np.float32)
    in3    = np.stack([inside] * 3, axis=2)
    room_f  = np.array(room).astype(np.float32)
    warped_f = warped.astype(np.float32)

    # ── 光影融合 ──────────────────────────────────────────
    # 牆面亮度圖（0–1）
    wall_lum = (0.299 * room_f[..., 0] +
                0.587 * room_f[..., 1] +
                0.114 * room_f[..., 2]) / 255.0

    # 字畫區域的牆面平均亮度（作為基準）
    mask_bool = inside > 0.5
    avg_lum   = float(np.mean(wall_lum[mask_bool])) if mask_bool.any() else 0.78
    avg_lum   = max(avg_lum, 0.1)

    # 亮度調整因子：讓字畫跟隨牆面明暗分布
    lum_factor = wall_lum / avg_lum
    lum_factor = cv2.GaussianBlur(lum_factor.astype(np.float32), (0, 0), sigmaX=60)
    lum_factor = np.clip(lum_factor, 0.55, 1.45)
    lum3 = np.stack([lum_factor] * 3, axis=2)

    # 光影版字畫 = 字畫顏色 × 牆面亮度因子
    warped_lit = np.clip(warped_f * lum3, 0, 255)

    # 混合：直接貼上（清晰）+ 光影版（融入感）
    warped_blend = warped_f * (1 - blend) + warped_lit * blend

    # 合成
    result = room_f * (1 - in3) + warped_blend * in3

    # ── 暖色立體陰影 ─────────────────────────────────────
    shift  = max(8, int(min(rw, rh) * 0.007))
    sigma  = shift * 2.2
    shadow_raw = cv2.GaussianBlur(warped_mask.astype(np.float32), (0, 0), sigmaX=sigma)
    shadow_s   = np.zeros_like(shadow_raw)
    shadow_s[shift:, shift:] = shadow_raw[:-shift, :-shift]
    shadow_s[warped_mask > 64] = 0
    alpha = shadow_s / 255.0 * 0.50

    # 深棕暖色陰影（R 保留最多 → 棕色調）
    result[..., 0] = np.clip(result[..., 0] * (1 - alpha * 0.70), 0, 255)
    result[..., 1] = np.clip(result[..., 1] * (1 - alpha * 0.85), 0, 255)
    result[..., 2] = np.clip(result[..., 2] * (1 - alpha * 1.00), 0, 255)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), 'RGB')


# ── 輔助函式 ──────────────────────────────────────────────

@st.cache_data
def load_catalog():
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, encoding='utf-8') as f:
        return json.load(f)


def list_rooms():
    return sorted(ROOMS_DIR.glob("*.jpg")) + sorted(ROOMS_DIR.glob("*.png"))


def compute_corners(tl, tr, aspect):
    """由左上、右上和比例，算出右下、左下"""
    x1, y1 = tl
    x2, y2 = tr
    dx, dy  = x2-x1, y2-y1
    w = (dx*dx + dy*dy) ** 0.5
    if w < 1:
        return None
    h = w * aspect
    px, py = -dy/w*h, dx/w*h
    if py < 0:
        px, py = -px, -py
    return [tl, tr, (x2+px, y2+py), (x1+px, y1+py)]


_SLIDER_KEYS = ["adj_dx", "adj_dy", "adj_scale", "adj_blend",
                "c0_dx", "c0_dy", "c1_dx", "c1_dy",
                "c2_dx", "c2_dy", "c3_dx", "c3_dy"]

def _reset_sliders():
    for k in _SLIDER_KEYS:
        st.session_state.pop(k, None)


def apply_adjustments(base_corners, dx, dy, scale_pct):
    """對 base_corners 套用左右/上下偏移和大小縮放，回傳調整後的 4 個角落"""
    scale = scale_pct / 100.0
    cx = sum(p[0] for p in base_corners) / 4
    cy = sum(p[1] for p in base_corners) / 4
    return [
        (cx + (x - cx) * scale + dx,
         cy + (y - cy) * scale + dy)
        for x, y in base_corners
    ]


def draw_corners(img_pil, corners, disp_scale):
    """在顯示圖上畫出角落標記和四邊形輪廓"""
    overlay = img_pil.copy().convert("RGB")
    draw    = ImageDraw.Draw(overlay)
    colors  = ["#e74c3c", "#27ae60", "#2980b9", "#f39c12"]

    disp = [(int(x * disp_scale), int(y * disp_scale)) for x, y in corners]

    if len(disp) == 4:
        for i in range(4):
            draw.line([disp[i], disp[(i+1) % 4]], fill="white", width=2)

    r = 10
    for i, (cx, cy) in enumerate(disp):
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=colors[i], outline="white")
        draw.text((cx-4, cy-7), str(i+1), fill="white")

    return overlay


# ── Streamlit App ──────────────────────────────────────────

st.set_page_config(
    page_title="涵晞草堂 · 字畫展示",
    layout="wide"
)

st.title("涵晞草堂 · 字畫展示工具")
st.caption("將書法字畫合成到室內裝潢照片，展示掛放效果")

catalog = load_catalog()
if not catalog:
    st.error("找不到字畫資料，請確認 artworks_nobg/ 資料夾存在。")
    st.stop()

# ── Session state 初始化 ──
for key, val in [("base_corners", []), ("last_click", None),
                 ("result_img", None), ("art_key", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── 側邊欄：選字畫 ──────────────────────────────────────
with st.sidebar:
    st.header("選擇字畫")

    categories = list(catalog.keys())
    category   = st.selectbox("類別", categories)
    items      = catalog[category]
    art_names  = [item['filename'] for item in items]
    art_idx    = st.selectbox("作品", range(len(art_names)),
                               format_func=lambda i: art_names[i])
    selected   = items[art_idx]
    # 使用相對路徑，避免依賴 catalog 裡的絕對路徑
    art_path   = WEB_DIR / selected['category'] / selected['filename']

    # 切換字畫時自動重置
    art_key = f"{category}/{art_idx}"
    if st.session_state.art_key != art_key:
        st.session_state.art_key      = art_key
        st.session_state.base_corners = []
        st.session_state.last_click   = None
        st.session_state.result_img   = None
        _reset_sliders()

    if art_path.exists():
        thumb = Image.open(art_path)
        thumb.thumbnail((220, 320), Image.LANCZOS)
        st.image(thumb, caption=selected['filename'], use_container_width=True)
    else:
        st.warning("找不到字畫檔案")

    st.divider()
    if st.button("重新選取位置", use_container_width=True):
        st.session_state.base_corners = []
        st.session_state.last_click   = None
        st.session_state.result_img   = None
        _reset_sliders()
        st.rerun()

# ── 主區域 ────────────────────────────────────────────────

# 選室內圖
st.subheader("步驟 1：選擇室內圖")
rooms = list_rooms()
room_img = None

col_sel, col_up = st.columns([1, 1])
with col_sel:
    if rooms:
        room_names = [r.name for r in rooms]
        sel_name   = st.selectbox("從現有室內圖選擇", room_names)
        room_img   = Image.open(ROOMS_DIR / sel_name).convert("RGB")

with col_up:
    uploaded = st.file_uploader("或上傳自訂室內圖", type=["jpg","jpeg","png"])
    if uploaded:
        room_img = Image.open(uploaded).convert("RGB")

if room_img is None:
    st.info("請選擇或上傳室內圖")
    st.stop()

# 計算字畫長寬比
art_pil = None
if art_path.exists():
    art_pil = Image.open(art_path).convert("RGB")
    _, _, _, (tw, th) = build_mounted_artwork(art_pil)
    aspect = th / tw
else:
    aspect = 1.5

# 互動點選區
st.subheader("步驟 2：點選字畫位置")

n = len(st.session_state.base_corners)
if n == 0:
    st.info("在圖片上點選字畫的 **左上角**")
elif n == 1:
    st.info("再點選字畫的 **右上角**（下方兩角自動依比例產生）")
elif n == 4:
    st.success("位置已設定，用下方滑桿微調後按「開始合成」")

# 顯示比例（原圖縮小到最多 900px 寬供操作）
MAX_W      = 900
rw, rh     = room_img.size
disp_scale = min(MAX_W / rw, 1.0)
disp_img   = room_img.resize((int(rw * disp_scale), int(rh * disp_scale)), Image.LANCZOS)

# 套用滑桿調整，取得實際使用的 corners
final_corners = None
if len(st.session_state.base_corners) == 4:
    # 整體調整
    overall = apply_adjustments(
        st.session_state.base_corners,
        st.session_state.get("adj_dx", 0),
        st.session_state.get("adj_dy", 0),
        st.session_state.get("adj_scale", 100),
    )
    # 個別角落偏移（疊加在整體調整上）
    final_corners = [
        (x + st.session_state.get(f"c{i}_dx", 0),
         y + st.session_state.get(f"c{i}_dy", 0))
        for i, (x, y) in enumerate(overall)
    ]
    disp_img = draw_corners(disp_img, final_corners, disp_scale)
elif st.session_state.base_corners:
    disp_img = draw_corners(disp_img, st.session_state.base_corners, disp_scale)

# 圖片點選元件
click = streamlit_image_coordinates(disp_img, key="room_click")

if click is not None:
    pos = (click["x"], click["y"])
    if pos != st.session_state.last_click:
        st.session_state.last_click = pos
        x_orig = int(click["x"] / disp_scale)
        y_orig = int(click["y"] / disp_scale)
        n = len(st.session_state.base_corners)

        if n == 0:
            st.session_state.base_corners = [(x_orig, y_orig)]
        elif n == 1:
            full = compute_corners(st.session_state.base_corners[0], (x_orig, y_orig), aspect)
            if full:
                st.session_state.base_corners = full
                st.session_state.result_img   = None
                _reset_sliders()
        else:
            # 再點一下重置
            st.session_state.base_corners = [(x_orig, y_orig)]
            st.session_state.result_img   = None
            _reset_sliders()
        st.rerun()

# 滑桿微調（4 個角落已設定才顯示）
if len(st.session_state.base_corners) == 4:

    # ── 整體調整 ──────────────────────────────────────────
    st.markdown("**整體調整**")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.slider("左右移動", -300, 300, 0, step=5, key="adj_dx",
                  help="整體向左／向右平移")
    with g2:
        st.slider("上下移動", -300, 300, 0, step=5, key="adj_dy",
                  help="整體向上／向下平移")
    with g3:
        st.slider("大小 (%)", 50, 200, 100, step=5, key="adj_scale",
                  help="100 為原始大小")
    with g4:
        st.slider("光影融入 (%)", 0, 100, 45, step=5, key="adj_blend",
                  help="0=直接貼上  100=完全跟隨牆面光影")

    # ── 個別角落微調 ──────────────────────────────────────
    with st.expander("個別角落微調（透視校正）"):
        st.caption("可單獨移動某個角落，修正牆面透視變形")
        corner_labels = ["① 左上", "② 右上", "③ 右下", "④ 左下"]
        cols = st.columns(4)
        for i, label in enumerate(corner_labels):
            with cols[i]:
                st.markdown(f"**{label}**")
                st.slider("← →", -150, 150, 0, step=3,
                          key=f"c{i}_dx", label_visibility="collapsed",
                          help=f"{label} 左右移動")
                st.slider("↑ ↓", -150, 150, 0, step=3,
                          key=f"c{i}_dy", label_visibility="collapsed",
                          help=f"{label} 上下移動")

    # 合成按鈕
    st.divider()
    if st.button("開始合成", type="primary", use_container_width=True):
        if art_pil is None:
            st.error("找不到字畫檔案")
        elif final_corners is None:
            st.error("請先設定位置")
        else:
            with st.spinner("合成中，請稍候..."):
                blend_val = st.session_state.get("adj_blend", 45) / 100.0
                result = composite_artwork(room_img, art_pil, final_corners,
                                           blend=blend_val)
                st.session_state.result_img = result

# 顯示結果
if st.session_state.result_img is not None:
    st.divider()
    st.subheader("合成結果")
    st.image(st.session_state.result_img, use_container_width=True)

    buf = _io.BytesIO()
    st.session_state.result_img.save(buf, format="JPEG", quality=92)
    st.download_button(
        label="下載成品 (JPG)",
        data=buf.getvalue(),
        file_name=f"展示_{selected['filename']}.jpg",
        mime="image/jpeg",
        use_container_width=True,
        type="primary"
    )
