"""Motion-графіка: процедурно малює кожен кадр (PIL + numpy) і віддає в ffmpeg."""
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from timeline import FPS, H, HERE, SCENES, TOTAL, W

FONT_DIR = "/usr/share/fonts/opentype/inter/"

# ---------- палітра ----------
PINK = (255, 46, 136)
HOT = (255, 62, 165)
ORANGE = (255, 154, 60)
YELLOW = (255, 214, 80)
CYAN = (45, 226, 230)
PURPLE = (40, 8, 70)
DEEP = (14, 3, 30)
WHITE = (255, 255, 255)
RED = (255, 50, 70)


@lru_cache(None)
def font(size, weight="Black", italic=False, display=True):
    name = ("InterDisplay-" if display else "Inter-") + weight + ("Italic" if italic else "") + ".otf"
    return ImageFont.truetype(FONT_DIR + name, size)


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, t0, d):
    return clamp((t - t0) / d)


def out_cubic(x):
    return 1 - (1 - x) ** 3


def out_expo(x):
    return 1 if x >= 1 else 1 - 2 ** (-10 * x)


def in_out(x):
    return 3 * x * x - 2 * x * x * x


def out_back(x, s=1.7):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def lerp(a, b, x):
    return a + (b - a) * x


def mix_col(c1, c2, x):
    return tuple(int(lerp(a, b, x)) for a, b in zip(c1, c2))


# ---------- статичні шари (обчислюються один раз на процес) ----------
HORIZON = int(H * 0.64)


def vgrad(h, w, stops):
    """Вертикальний градієнт: stops = [(pos, color), ...]"""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        col = np.interp(ys, [p for p, _ in stops], [s[c] for _, s in stops])
        out[:, :, c] = col[:, None]
    return out


def make_palm(height, lean, seed, color=(16, 2, 26)):
    r = np.random.default_rng(seed)
    w_img, h_img = int(height * 1.6), int(height * 1.25)
    img = Image.new("L", (w_img, h_img), 0)
    d = ImageDraw.Draw(img)
    bx, by = w_img // 2, h_img - 2
    pts_l, pts_r = [], []
    steps = 40
    top = None
    for k in range(steps + 1):
        s = k / steps
        x = bx + lean * height * s * s
        y = by - s * height
        wdt = lerp(height * 0.045, height * 0.022, s)
        pts_l.append((x - wdt, y))
        pts_r.append((x + wdt, y))
        top = (x, y)
    d.polygon(pts_l + pts_r[::-1], fill=255)
    # кільця на стовбурі
    for k in range(4, steps, 2):
        s = k / steps
        x = bx + lean * height * s * s
        y = by - s * height
        wdt = lerp(height * 0.05, height * 0.026, s)
        d.line([(x - wdt, y), (x + wdt, y + 2)], fill=255, width=3)
    tx, ty = top
    nf = 9
    for f in range(nf):
        ang = -math.pi / 2 + (f - (nf - 1) / 2) * (math.pi * 1.05 / (nf - 1)) + r.uniform(-0.12, 0.12)
        length = height * r.uniform(0.42, 0.6)
        droop = r.uniform(0.9, 1.4)
        cl = []
        for k in range(30):
            s = k / 29
            x = tx + math.cos(ang) * length * s
            y = ty + math.sin(ang) * length * s + droop * length * 0.55 * s * s
            cl.append((x, y))
        left, right = [], []
        for k in range(30):
            s = k / 29
            x0, y0 = cl[max(0, k - 1)]
            x1, y1 = cl[min(29, k + 1)]
            nx, ny = -(y1 - y0), (x1 - x0)
            nl = math.hypot(nx, ny) or 1
            wdt = height * 0.05 * math.sin(math.pi * min(1, s * 1.15)) ** 0.8
            # зубчики листя
            if k % 3 == 1:
                wdt *= 1.35
            left.append((cl[k][0] + nx / nl * wdt, cl[k][1] + ny / nl * wdt))
            right.append((cl[k][0] - nx / nl * wdt * 0.6, cl[k][1] - ny / nl * wdt * 0.6))
        d.polygon(left + right[::-1], fill=255)
    d.ellipse([tx - height * 0.03, ty - height * 0.02, tx + height * 0.03, ty + height * 0.04], fill=255)
    rgba = Image.new("RGBA", img.size, color + (0,))
    rgba.putalpha(img)
    return rgba, (bx, by)


class Assets:
    def __init__(self):
        r = np.random.default_rng(3)
        # небо на заході сонця
        self.sky = vgrad(HORIZON, W, [(0, (12, 2, 40)), (0.35, (70, 10, 100)), (0.7, (220, 40, 120)), (1, (255, 140, 70))])
        # нічне небо (для сцен зі статистикою)
        self.night = vgrad(HORIZON, W, [(0, (6, 1, 18)), (0.6, (30, 6, 60)), (1, (90, 15, 110))])
        # підлога
        self.floor = vgrad(H - HORIZON, W, [(0, (40, 5, 60)), (1, (8, 1, 18))])
        # сонце
        sr = 300
        sun = Image.new("RGBA", (sr * 2, sr * 2), (0, 0, 0, 0))
        g = vgrad(sr * 2, sr * 2, [(0, YELLOW), (0.55, ORANGE), (1, HOT)])
        gimg = Image.fromarray(g.astype(np.uint8))
        m = Image.new("L", (sr * 2, sr * 2), 0)
        md = ImageDraw.Draw(m)
        md.ellipse([0, 0, sr * 2, sr * 2], fill=255)
        for k in range(9):  # смуги ретро-сонця
            y = sr * 1.0 + k * 34
            hgt = 4 + k * 3.2
            md.rectangle([0, y, sr * 2, y + hgt], fill=0)
        sun = gimg.convert("RGBA")
        sun.putalpha(m)
        self.sun = sun
        glow = Image.new("RGBA", (sr * 4, sr * 4), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse([sr * 0.8, sr * 0.8, sr * 3.2, sr * 3.2], fill=(255, 90, 120, 130))
        self.sun_glow = glow.filter(ImageFilter.GaussianBlur(120))
        # силует міста
        city = Image.new("RGBA", (W + 600, 380), (0, 0, 0, 0))
        cd = ImageDraw.Draw(city)
        x = 0
        while x < W + 600:
            bw = int(r.integers(40, 130))
            bh = int(r.integers(60, 340))
            if r.random() < 0.08:
                bh = int(r.integers(300, 375))
            col = (34, 6, 52, 255)
            cd.rectangle([x, 380 - bh, x + bw, 380], fill=col)
            if r.random() < 0.3:
                cd.rectangle([x + bw // 2 - 2, 380 - bh - 30, x + bw // 2 + 2, 380 - bh], fill=col)
            for wy in range(380 - bh + 12, 370, 16):
                for wx in range(x + 8, x + bw - 8, 14):
                    if r.random() < 0.18:
                        c = (255, 200, 120, 200) if r.random() < 0.7 else (120, 230, 255, 200)
                        cd.rectangle([wx, wy, wx + 4, wy + 6], fill=c)
            x += bw + int(r.integers(0, 12))
        self.city = city
        # пальми (передній план)
        self.palms = []
        for seed, hgt, lean, px, py in [(1, 720, 0.28, -60, H + 20), (2, 560, -0.22, 260, H + 60),
                                         (3, 760, -0.3, W - 120, H + 30), (4, 520, 0.25, W - 420, H + 80)]:
            img, base = make_palm(hgt, lean, seed)
            self.palms.append((img, base, px, py))
        # зірки
        self.stars = r.random((160, 3)) * np.array([W, HORIZON * 0.8, 1])
        # віньєтка
        yy, xx = np.mgrid[0:H, 0:W]
        dd = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.45 * dd ** 2, 0.25, 1).astype(np.float32)[:, :, None]
        self.grain = [r.normal(0, 7, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]


A = None


def assets():
    global A
    if A is None:
        A = Assets()
    return A


# ---------- шари фону ----------
def background(t, mode="sunset", sun_y=0.0, grid_speed=1.0, palms=True, city=True, tint=None):
    a = assets()
    arr = np.empty((H, W, 3), np.float32)
    arr[:HORIZON] = a.sky if mode == "sunset" else a.night
    arr[HORIZON:] = a.floor
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    # зірки
    for x, y, s in a.stars:
        tw = 0.5 + 0.5 * math.sin(t * 2 + s * 20)
        v = int(120 + 120 * tw * s)
        rr = 1 + (s > 0.85)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(v, v, min(255, v + 30)))
    # сонце
    if mode == "sunset":
        cx, cy = W // 2, int(HORIZON - 170 + sun_y)
        img.paste(a.sun_glow, (cx - a.sun_glow.width // 2, cy - a.sun_glow.height // 2), a.sun_glow)
        img.paste(a.sun, (cx - a.sun.width // 2, cy - a.sun.height // 2), a.sun)
    if city:
        off = int((t * 12) % 600)
        img.paste(a.city, (-off, HORIZON - a.city.height + 2), a.city)
    # неонова сітка
    grid = Image.new("RGB", (W // 2, (H - HORIZON) // 2), (0, 0, 0))
    gd = ImageDraw.Draw(grid)
    gw, gh = grid.size
    vx = gw / 2
    for k in range(-22, 23):
        gd.line([(vx + k * 6, 0), (vx + k * 95, gh)], fill=HOT, width=1)
    phase = (t * 0.9 * grid_speed) % 1.0
    for k in range(14):
        z = (k + 1 - phase)
        if z <= 0:
            continue
        y = gh * (0.12 / z) * 3.2
        if y < gh:
            gd.line([(0, y), (gw, y)], fill=HOT, width=1)
    gd.line([(0, 0), (gw, 0)], fill=(255, 150, 200), width=1)
    glow = grid.filter(ImageFilter.GaussianBlur(3))
    grid = ImageChops.add(grid, glow)
    grid = grid.resize((W, H - HORIZON), Image.BILINEAR)
    region = img.crop((0, HORIZON, W, H))
    img.paste(ImageChops.add(region, grid), (0, HORIZON))
    if palms:
        for k, (p, (bx, by), px, py) in enumerate(a.palms):
            sway = math.sin(t * 0.8 + k) * 6
            img.paste(p, (int(px - bx + sway), int(py - by)), p)
    if tint:
        col, amt = tint
        img = Image.blend(img, Image.new("RGB", img.size, col), amt)
    return img


# ---------- текст ----------
def text_layer(text, fnt, fill=WHITE, grad=None, tracking=0, letter_fx=None, stroke=0, stroke_fill=None):
    asc, desc = fnt.getmetrics()
    h = asc + desc
    total = fnt.getlength(text) + tracking * max(0, len(text) - 1)
    pad = int(h * 0.5) + stroke
    lw, lh = int(total) + 2 * pad, h + 2 * pad
    mask = Image.new("L", (lw, lh), 0)
    d = ImageDraw.Draw(mask)
    for i, ch in enumerate(text):
        x = pad + fnt.getlength(text[:i]) + tracking * i
        dy, al = letter_fx(i, len(text)) if letter_fx else (0, 1)
        if al <= 0.01 or ch == " ":
            continue
        d.text((x, pad + dy), ch, font=fnt, fill=int(255 * clamp(al)), stroke_width=stroke, stroke_fill=int(255 * clamp(al)))
    if grad:
        g = vgrad(lh, lw, [(0.25, grad[0]), (0.75, grad[1])]).astype(np.uint8)
        col = Image.fromarray(g).convert("RGBA")
    else:
        col = Image.new("RGBA", (lw, lh), fill + (255,))
    col.putalpha(mask)
    return col


def place(img, layer, cx, cy, alpha=1.0, scale=1.0, glow=None, glow_amt=1.0, rot=0, anchor="mm"):
    """Ставить шар на кадр. glow=(колір, радіус) — неонове світіння (адитивне)."""
    if alpha <= 0.01:
        return
    if scale != 1.0:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BILINEAR)
    if rot:
        layer = layer.rotate(rot, Image.BILINEAR, expand=True)
    if alpha < 1:
        al = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer = layer.copy()
        layer.putalpha(al)
    if anchor == "mm":
        x, y = int(cx - layer.width / 2), int(cy - layer.height / 2)
    elif anchor == "lm":
        x, y = int(cx), int(cy - layer.height / 2)
    else:
        x, y = int(cx - layer.width), int(cy - layer.height / 2)
    if glow:
        gcol, rad = glow
        pad = int(rad * 2.2)
        A_ = Image.new("L", (layer.width + 2 * pad, layer.height + 2 * pad), 0)
        A_.paste(layer.getchannel("A"), (pad, pad))
        small = A_.resize((max(1, A_.width // 4), max(1, A_.height // 4)), Image.BILINEAR)
        small = small.filter(ImageFilter.GaussianBlur(max(1, rad / 4)))
        gm = small.resize(A_.size, Image.BILINEAR).point(lambda v: int(min(255, v * 0.95 * glow_amt)))
        gl = Image.new("RGB", A_.size, gcol)
        gl = ImageChops.multiply(gl, Image.merge("RGB", (gm, gm, gm)))
        _add_clip(img, gl, x - pad, y - pad)
        # темний контур для читабельності на фоні світіння
        sh = layer.getchannel("A").filter(ImageFilter.MaxFilter(5)).point(lambda v: int(v * 0.55))
        img.paste((12, 2, 26), (x, y), sh) if x >= 0 and y >= 0 else _paste_neg(img, (12, 2, 26), x, y, sh)
    if x >= 0 and y >= 0:
        img.paste(layer, (x, y), layer)
    else:
        _paste_neg(img, layer, x, y, layer)


def _paste_neg(img, src, x, y, mask):
    """paste з від'ємними координатами (обрізаємо джерело)."""
    cx, cy = max(0, -x), max(0, -y)
    m = mask.crop((cx, cy, mask.width, mask.height))
    if isinstance(src, tuple):
        img.paste(src, (x + cx, y + cy, x + cx + m.width, y + cy + m.height), m)
    else:
        img.paste(src.crop((cx, cy, src.width, src.height)), (x + cx, y + cy), m)


def _add_clip(img, rgb, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + rgb.width), min(H, y + rgb.height)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgb.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    reg = img.crop((x0, y0, x1, y1))
    img.paste(ImageChops.add(reg, sub), (x0, y0))


def txt(img, text, size, cx, cy, t=None, t0=0, weight="Black", italic=False, fill=WHITE, grad=None, glow=None,
        tracking=0, alpha=1.0, scale=1.0, mode="rise", stagger=0.035, dur=0.5, anchor="mm", glow_amt=1.0, rot=0):
    """Текст з анімацією появи: rise (по буквах знизу), type (друк), fade."""
    fnt = font(size, weight, italic)
    fx = None
    if t is not None and mode == "rise":
        def fx(i, n):
            p = out_cubic(prog(t, t0 + i * stagger, dur))
            return (1 - p) * size * 0.6, p
    elif t is not None and mode == "type":
        def fx(i, n):
            return 0, 1.0 if (t - t0) > i * stagger else 0.0
    elif t is not None and mode == "fade":
        alpha *= out_cubic(prog(t, t0, dur))
    layer = text_layer(text, fnt, fill, grad, tracking, fx)
    place(img, layer, cx, cy, alpha, scale, glow, glow_amt, rot, anchor)


def fmt(n):
    return f"{int(n):,}".replace(",", " ")


def rounded(d, box, r, **kw):
    d.rounded_rectangle(box, r, **kw)


def overlay_panel(img, box, alpha=0.55, radius=28, border=HOT):
    """Напівпрозора "скляна" плашка з неоновою рамкою."""
    x0, y0, x1, y1 = map(int, box)
    reg = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(12))
    dark = Image.new("RGB", reg.size, (10, 2, 22))
    reg = Image.blend(reg, dark, alpha)
    m = Image.new("L", reg.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, reg.width - 1, reg.height - 1], radius, fill=255)
    img.paste(reg, (x0, y0), m)
    if border:
        lay = Image.new("RGBA", (x1 - x0 + 80, y1 - y0 + 80), (0, 0, 0, 0))
        ImageDraw.Draw(lay).rounded_rectangle([40, 40, 40 + x1 - x0, 40 + y1 - y0], radius, outline=border + (255,), width=3)
        place(img, lay, (x0 + x1) / 2, (y0 + y1) / 2, glow=(border, 18))


def neon_shape(img, draw_fn, size, cx, cy, color=HOT, glow_r=22, alpha=1.0, scale=1.0, rot=0):
    lay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(lay), size)
    place(img, lay, cx, cy, alpha, scale, (color, glow_r), rot=rot)


# ---------- сцени ----------
def scene0(t, D):  # Інтро
    img = background(t, "sunset", sun_y=lerp(160, 0, out_cubic(prog(t, 0, 3))))
    # тонування для читабельності
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.18)
    p = out_back(prog(t, 0.25, 0.7))
    txt(img, "ЧОМУ", 64, W / 2, 210, t, 0.1, weight="ExtraBold", tracking=24, fill=CYAN, glow=(CYAN, 20))
    sc = lerp(2.4, 1.0, out_expo(prog(t, 0.25, 0.6)))
    al = prog(t, 0.25, 0.25)
    fnt = font(300, "Black", True)
    lay = text_layer("GTA VI", fnt, grad=(YELLOW, HOT), tracking=6)
    pulse = 1 + 0.015 * math.sin(t * 3)
    place(img, lay, W / 2, 420, al, sc * pulse, (HOT, 40), glow_amt=1.2)
    # підзаголовок + лінія
    txt(img, "НАЙОЧІКУВАНІША ГРА", 88, W / 2, 640, t, 1.0, tracking=4, glow=(PINK, 18))
    txt(img, "У СВІТІ", 88, W / 2, 745, t, 1.35, tracking=10, grad=(CYAN, (140, 120, 255)), glow=(CYAN, 18))
    lp_ = out_cubic(prog(t, 1.6, 0.8))
    d = ImageDraw.Draw(img)
    if lp_ > 0:
        wdt = 720 * lp_
        d.rectangle([W / 2 - wdt / 2, 812, W / 2 + wdt / 2, 818], fill=YELLOW)
    # спалах на старті
    fl = 1 - prog(t, 0.25, 0.5)
    if 0 < fl < 1 and t > 0.25:
        img = Image.blend(img, Image.new("RGB", img.size, (255, 230, 240)), 0.6 * fl)
    return img


def scene1(t, D):  # 13 років
    img = background(t, "night", palms=False, city=True)
    yr = int(lerp(2013, 2026, in_out(prog(t, 0.4, 3.2))))
    txt(img, "GTA V ВИЙШЛА У", 52, W / 2, 170, t, 0.1, weight="Bold", tracking=8, fill=(220, 200, 255))
    txt(img, str(yr), 300, W / 2, 380, alpha=prog(t, 0.2, 0.3), grad=(YELLOW, HOT), glow=(HOT, 36), italic=True)
    # таймлайн
    d = ImageDraw.Draw(img)
    x0, x1, y = 260, W - 260, 640
    pp = in_out(prog(t, 0.4, 3.2))
    d.rounded_rectangle([x0, y - 6, x1, y + 6], 6, fill=(60, 20, 90))
    bar = Image.new("RGBA", (int((x1 - x0) * pp) + 1, 12), HOT + (255,))
    place(img, bar, x0, y, glow=(HOT, 16), anchor="lm")
    for k, yy in enumerate(range(2013, 2027)):
        x = x0 + (x1 - x0) * k / 13
        on = pp >= k / 13 - 1e-6
        r = 10 if yy in (2013, 2026) else 6
        d.ellipse([x - r, y - r, x + r, y + r], fill=YELLOW if on else (90, 40, 120))
        if yy in (2013, 2026):
            txt(img, str(yy), 40, x, y + 52, weight="Bold", fill=WHITE if on else (150, 120, 180))
    # велике "13 РОКІВ"
    s = out_back(prog(t, 3.7, 0.6))
    if s > 0:
        txt(img, "13+ РОКІВ ОЧІКУВАННЯ", 110, W / 2, 840, scale=lerp(0.6, 1, s), alpha=prog(t, 3.7, 0.2),
            grad=(CYAN, (150, 110, 255)), glow=(CYAN, 30))
    txt(img, "ціле покоління гравців виросло, чекаючи", 40, W / 2, 950, t, 5.0, mode="fade", dur=0.6,
        weight="Medium", fill=(230, 210, 255))
    return img


def scene2(t, D):  # продажі
    img = background(t, "night", palms=False, city=False, grid_speed=0.6)
    d = ImageDraw.Draw(img)
    # стовпчики-"коробки" на фоні
    n = 14
    for k in range(n):
        p = out_cubic(prog(t, 0.3 + k * 0.08, 1.2))
        hgt = (120 + 460 * ((k + 1) / n) ** 1.6) * p
        x = 140 + k * 120
        col = mix_col((70, 20, 110), HOT, k / n)
        d.rounded_rectangle([x, HORIZON - hgt, x + 80, HORIZON], 8, fill=col + (255,))
        d.rectangle([x, HORIZON - hgt, x + 80, HORIZON - hgt + 6], fill=YELLOW)
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.35)
    overlay_panel(img, (360, 180, W - 360, 620), alpha=0.6)
    v = 200_000_000 * out_expo(prog(t, 0.5, 3.0))
    txt(img, "ПРОДАНО КОПІЙ GTA V", 50, W / 2, 255, t, 0.3, weight="Bold", tracking=8, fill=CYAN, glow=(CYAN, 12))
    txt(img, fmt(v) + ("+" if v > 199_000_000 else ""), 170, W / 2, 410, grad=(YELLOW, ORANGE), glow=(ORANGE, 30), italic=True)
    txt(img, "понад 200 мільйонів", 46, W / 2, 545, t, 2.2, mode="fade", weight="SemiBold", fill=(240, 220, 255))
    # бейдж
    s = out_back(prog(t, 4.0, 0.5))
    if s > 0:
        b = Image.new("RGBA", (1100, 120), (0, 0, 0, 0))
        bd = ImageDraw.Draw(b)
        bd.rounded_rectangle([0, 0, 1099, 119], 60, fill=HOT + (255,))
        lay = text_layer("№2 НАЙПРОДАВАНІША ГРА В ІСТОРІЇ", font(52, "Black"), WHITE)
        b.paste(lay, ((1100 - lay.width) // 2, (120 - lay.height) // 2), lay)
        place(img, b, W / 2, 780, scale=s, glow=(HOT, 30))
    txt(img, "очікування від нової частини — колосальні", 44, W / 2, 920, t, 6.5, mode="fade", weight="Medium",
        fill=(230, 210, 255))
    return img


def play_icon(d, size):
    w, h = size
    d.ellipse([20, 20, w - 20, h - 20], outline=WHITE + (255,), width=14)
    cx, cy = w / 2, h / 2
    r = w * 0.2
    d.polygon([(cx - r * 0.7, cy - r), (cx - r * 0.7, cy + r), (cx + r, cy)], fill=WHITE + (255,))


def stamp(img, text, cx, cy, t, t0, color=RED, size=60, rot=-8):
    p = prog(t, t0, 0.35)
    if p <= 0:
        return
    s = lerp(2.2, 1.0, out_cubic(p))
    fnt = font(size, "Black")
    lay = text_layer(text, fnt, color, tracking=4)
    box = Image.new("RGBA", (lay.width + 20, lay.height), (0, 0, 0, 0))
    bd = ImageDraw.Draw(box)
    bd.rounded_rectangle([int(size * 0.3), int(size * 0.35), box.width - int(size * 0.3), box.height - int(size * 0.35)],
                         12, outline=color + (255,), width=8)
    box.paste(lay, (10, 0), lay)
    place(img, box, cx, cy, alpha=p, scale=s, rot=rot, glow=(color, 14))


def scene3(t, D):  # трейлер 1
    img = background(t, "sunset", sun_y=60, palms=True)
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.45)
    s = out_back(prog(t, 0.1, 0.6)) * (1 + 0.04 * math.sin(t * 5))
    neon_shape(img, play_icon, (300, 300), 430, 470, color=HOT, scale=s, glow_r=40)
    txt(img, "ТРЕЙЛЕР №1", 46, 430, 680, t, 0.3, weight="Bold", tracking=10, fill=CYAN, glow=(CYAN, 12))
    txt(img, "ГРУДЕНЬ 2023", 40, 430, 740, t, 0.5, weight="Medium", fill=(230, 210, 255))
    v = 90_000_000 * out_expo(prog(t, 0.5, 2.4))
    txt(img, fmt(v), 150, 1260, 400, grad=(YELLOW, HOT), glow=(HOT, 28), italic=True)
    txt(img, "ПЕРЕГЛЯДІВ НА YOUTUBE", 56, 1260, 525, t, 0.8, tracking=4)
    txt(img, "ЗА 24 ГОДИНИ", 56, 1260, 600, t, 1.1, tracking=4, fill=YELLOW, glow=(ORANGE, 14))
    stamp(img, "СВІТОВИЙ РЕКОРД", 1260, 770, t, 2.8, color=CYAN, size=64, rot=-6)
    txt(img, "найпопулярніше немузичне відео за першу добу", 38, W / 2, 960, t, 4.2, mode="fade", weight="Medium",
        fill=(230, 210, 255))
    return img


def scene4(t, D):  # трейлер 2
    img = background(t, "night", palms=False)
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.3)
    txt(img, "ТРЕЙЛЕР №2  ·  ТРАВЕНЬ 2025", 48, W / 2, 150, t, 0.1, weight="Bold", tracking=6, fill=CYAN, glow=(CYAN, 12))
    v = 475_000_000 * out_expo(prog(t, 0.4, 3.0))
    txt(img, fmt(v), 190, W / 2, 330, grad=(YELLOW, HOT), glow=(HOT, 36), italic=True)
    txt(img, "ПЕРЕГЛЯДІВ ЗА 24 ГОДИНИ НА ВСІХ ПЛАТФОРМАХ", 46, W / 2, 470, t, 0.8, weight="Bold", tracking=3)
    # порівняльні стовпчики
    d = ImageDraw.Draw(img)
    x0, maxw = 560, 1100
    rows = [("ТРЕЙЛЕР 1", 90, (150, 110, 255), 1.6), ("ТРЕЙЛЕР 2", 475, HOT, 2.0)]
    for k, (lab, val, col, st) in enumerate(rows):
        y = 640 + k * 150
        p = out_cubic(prog(t, st, 1.4))
        txt(img, lab, 44, x0 - 40, y, t, st - 0.2, weight="Bold", anchor="rm", mode="fade", dur=0.3)
        d.rounded_rectangle([x0, y - 40, x0 + maxw, y + 40], 14, fill=(35, 10, 60))
        wdt = max(1, int(maxw * val / 475 * p))
        bar = Image.new("RGBA", (wdt, 80), col + (255,))
        place(img, bar, x0, y, glow=(col, 20), anchor="lm", alpha=1 if p > 0 else 0)
        if p > 0.3:
            txt(img, f"{int(val * p)} млн", 42, x0 + wdt + 24, y, weight="Black", anchor="lm", fill=WHITE)
    s = out_back(prog(t, 4.2, 0.5))
    if s > 0:
        txt(img, "×5", 120, W - 190, 640, scale=s, grad=(YELLOW, ORANGE), glow=(ORANGE, 24), italic=True)
    return img


def person(d, size, hair=False, cap=False, col=HOT):
    w, h = size
    cx = w / 2
    c = col + (255,)
    if hair:  # довге волосся
        d.ellipse([cx - 105, 70, cx + 105, 300], fill=c)
        d.rectangle([cx - 105, 185, cx + 105, 330], fill=c)
    d.ellipse([cx - 80, 80, cx + 80, 250], fill=c)  # голова
    d.rectangle([cx - 30, 230, cx + 30, 290], fill=c)  # шия
    d.rounded_rectangle([cx - 180, 280, cx + 180, h + 80], 120, fill=c)  # плечі
    if cap:
        d.chord([cx - 88, 60, cx + 88, 200], 180, 360, fill=c)
        d.rounded_rectangle([cx - 10, 118, cx + 130, 140], 10, fill=c)


def char_card(img, name, cx, cy, t, t0, hair, cap, col, from_dir):
    p = out_cubic(prog(t, t0, 0.8))
    if p <= 0:
        return
    x = cx + from_dir * (1 - p) * 700
    cw, ch = 470, 600
    card = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    cd = ImageDraw.Draw(card)
    g = vgrad(ch, cw, [(0, (60, 12, 90)), (1, (15, 3, 30))]).astype(np.uint8)
    gi = Image.fromarray(g).convert("RGBA")
    m = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, cw - 1, ch - 1], 30, fill=255)
    card.paste(gi, (0, 0), m)
    sil = Image.new("RGBA", (cw, 480), (0, 0, 0, 0))
    person(ImageDraw.Draw(sil), (cw, 480), hair, cap, col)
    card.paste(sil, (0, 40), sil)
    cd.rounded_rectangle([0, 0, cw - 1, ch - 1], 30, outline=col + (255,), width=4)
    cd.rectangle([0, ch - 120, cw, ch - 30], fill=(10, 2, 20, 230))
    lay = text_layer(name, font(64, "Black"), WHITE, tracking=6)
    card.paste(lay, ((cw - lay.width) // 2, ch - 75 - lay.height // 2), lay)
    place(img, card, x, cy, alpha=p, glow=(col, 26))


def scene5(t, D):  # Вайс-Сіті, герої
    img = background(t, "sunset", sun_y=-10)
    p2 = prog(t, 4.6, 0.6)
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.15 + 0.45 * p2)
    # частина 1: місце дії
    a1 = 1 - p2
    if a1 > 0:
        txt(img, "ПОВЕРНЕННЯ У", 54, W / 2, 200, t, 0.1, weight="Bold", tracking=14, fill=WHITE, alpha=a1)
        txt(img, "VICE CITY", 230, W / 2, 360, t, 0.3, italic=True, grad=(CYAN, HOT), glow=(HOT, 40),
            alpha=a1, stagger=0.06, tracking=6)
        txt(img, "ШТАТ ЛЕОНІДА  ·  НАТХНЕННИЙ ФЛОРИДОЮ", 46, W / 2, 530, t, 1.4, mode="fade", weight="Bold",
            tracking=4, fill=YELLOW, glow=(ORANGE, 10), alpha=a1)
    # частина 2: герої
    if t > 4.6:
        txt(img, "ДВА ГОЛОВНІ ГЕРОЇ", 64, W / 2, 120, t, 4.7, tracking=10, fill=CYAN, glow=(CYAN, 16))
        char_card(img, "ДЖЕЙСОН", W / 2 - 330, 560, t, 5.0, False, True, CYAN, -1)
        char_card(img, "ЛУСІЯ", W / 2 + 330, 560, t, 5.7, True, False, HOT, 1)
        txt(img, "перша жінка-протагоніст у сучасній історії серії", 40, W / 2, 1000, t, 7.6, mode="fade",
            weight="SemiBold", fill=(255, 225, 240))
    return img


def scene6(t, D):  # перенесення
    img = background(t, "night", palms=False, city=True, grid_speed=0.4)
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.35)
    txt(img, "ДАТА РЕЛІЗУ", 60, W / 2, 140, t, 0.1, tracking=14, fill=CYAN, glow=(CYAN, 14))
    items = [("ОСІНЬ 2025", 0.4, 2.6), ("26 ТРАВНЯ 2026", 1.2, 5.0), ("19 ЛИСТОПАДА 2026", 6.0, None)]
    d = ImageDraw.Draw(img)
    for k, (lab, ta, tstrike) in enumerate(items):
        y = 330 + k * 210
        p = out_cubic(prog(t, ta, 0.6))
        if p <= 0:
            continue
        final = tstrike is None
        x = W / 2 - 120 + (1 - p) * -200
        col = YELLOW if final else WHITE
        size = 110 if final else 90
        txt(img, lab, size, x, y, alpha=p, grad=(YELLOW, HOT) if final else None, fill=col,
            glow=(HOT, 30) if final else None, italic=final)
        if tstrike is not None:
            sp = out_cubic(prog(t, tstrike, 0.35))
            if sp > 0:
                fnt = font(size, "Black")
                tw = fnt.getlength(lab)
                d.line([(x - tw / 2 - 20, y + 6), (x - tw / 2 - 20 + (tw + 40) * sp, y - 6)], fill=RED, width=12)
                if sp >= 1:
                    img_d = img
                    stamp(img_d, "ПЕРЕНЕСЕНО", x + tw / 2 + 230, y, t, tstrike + 0.35, color=RED, size=44, rot=-10)
                    d = ImageDraw.Draw(img)
        if final and p > 0.9:
            tw = font(size, "Black", True).getlength(lab)
            stamp(img, "ОФІЦІЙНО", x + tw / 2 + 190, y, t, 6.6, color=CYAN, size=44, rot=-8)
    txt(img, "кожне перенесення лише підігрівало ажіотаж", 42, W / 2, 990, t, 7.5, mode="fade", weight="Medium",
        fill=(230, 210, 255))
    return img


def scene7(t, D):  # фінал
    img = background(t, "sunset", sun_y=lerp(0, -40, prog(t, 0, D)))
    img = Image.blend(img, Image.new("RGB", img.size, DEEP), 0.3)
    s = lerp(1.8, 1.0, out_expo(prog(t, 0.4, 0.6)))
    txt(img, "19.11.2026", 230, W / 2, 300, alpha=prog(t, 0.4, 0.2), scale=s, grad=(YELLOW, HOT), glow=(HOT, 44),
        italic=True)
    # лічильник днів
    p = out_back(prog(t, 2.0, 0.6))
    if p > 0:
        overlay_panel(img, (W / 2 - 330, 470, W / 2 + 330, 740), alpha=0.6, border=CYAN)
        txt(img, "5", 200, W / 2 - 150, 600, scale=p * (1 + 0.05 * math.sin(t * 8)), grad=(CYAN, (140, 120, 255)),
            glow=(CYAN, 30))
        txt(img, "ДНІВ", 110, W / 2 + 90, 600, alpha=prog(t, 2.2, 0.3), tracking=6)
        txt(img, "ЗАЛИШИЛОСЬ", 34, W / 2, 500, t, 2.3, weight="Bold", tracking=12, fill=CYAN)
    txt(img, "PS5   ·   XBOX SERIES X|S", 48, W / 2, 820, t, 3.2, weight="Bold", tracking=8, fill=WHITE, mode="fade")
    txt(img, "А ВИ ГОТОВІ?", 90, W / 2, 950, t, 5.0, grad=(YELLOW, HOT), glow=(HOT, 26), italic=True)
    # затемнення в кінці
    fo = prog(t, D - 1.4, 1.4)
    if fo > 0:
        img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), fo)
    return img


SCENE_FN = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7]


# ---------- перехід: неонова діагональна шторка ----------
def wipe(img, t):
    for s in SCENES[1:]:
        dt = t - s["start"]
        if -0.35 <= dt <= 0.35:
            p = (dt + 0.35) / 0.7
            p = in_out(p)
            band = W * 1.6
            x = lerp(-band - 400, W + 400, p)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            sk = 400
            d.polygon([(x, 0), (x + band, 0), (x + band - sk, H), (x - sk, H)], fill=(20, 4, 40, 255))
            d.polygon([(x + band - 40, 0), (x + band, 0), (x + band - sk, H), (x + band - sk - 40, H)], fill=HOT + (255,))
            d.polygon([(x, 0), (x + 30, 0), (x + 30 - sk, H), (x - sk, H)], fill=CYAN + (255,))
            img.paste(lay, (0, 0), lay)
    return img


def frame(fi):
    t = fi / FPS
    idx = 0
    for k, s in enumerate(SCENES):
        if t >= s["start"]:
            idx = k
    s = SCENES[idx]
    img = SCENE_FN[idx](t - s["start"], s["dur"])
    img = wipe(img, t)
    # віньєтка + плівкове зерно
    a = assets()
    arr = np.asarray(img).astype(np.float32) * a.vignette
    g = a.grain[fi % 6]
    arr += np.repeat(np.repeat(g, 2, 0), 2, 1)
    # легка хроматична аберація по краях
    return np.clip(arr, 0, 255).astype(np.uint8).tobytes()


def main():
    out = os.path.join(HERE, "build", "video_noaudio.mp4")
    nframes = int(TOTAL * FPS)
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        # кадри-превʼю для перевірки
        os.makedirs(os.path.join(HERE, "build", "preview"), exist_ok=True)
        times = [float(x) for x in sys.argv[2:]]
        for tt in times:
            fi = int(tt * FPS)
            Image.frombytes("RGB", (W, H), frame(fi)).save(os.path.join(HERE, "build", "preview", f"f_{tt:06.2f}.jpg"), quality=85)
        return
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for k, buf in enumerate(pool.imap(frame, range(nframes), chunksize=4)):
            ff.stdin.write(buf)
            if k % 150 == 0:
                print(f"frame {k}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("done", out)


if __name__ == "__main__":
    main()
