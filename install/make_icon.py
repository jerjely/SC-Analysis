"""SC Analysis icon: rounded navy tile + one bright receptive-field target (RF map rings
with a spike-raster sweep through it). Writes PNG 512/256/small, .ico and .icns."""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
out = sys.argv[1]
S = 4 * 512
def lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
top, bot = (0x10, 0x1d, 0x3d), (0x06, 0x0b, 0x1a)
grad = Image.new('RGB', (S, S))
g = np.zeros((S, S, 3), np.uint8)
for y in range(S):
    g[y] = lerp(top, bot, y / (S - 1))
grad = Image.fromarray(g)
m = 16 * 4; r = 104 * 4
mask = Image.new('L', (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([m, m, S - m, S - m], r, fill=255)
img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
img.paste(grad, (0, 0), mask)
cx, cy = S // 2, int(S * 0.5)
# soft glow (RF hot spot)
glow = Image.new('RGBA', (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
for k, a in [(760, 40), (600, 60), (440, 90), (300, 120)]:
    gd.ellipse([cx - k, cy - k * 0.82, cx + k, cy + k * 0.82], fill=(255, 140, 70, a))
glow = glow.filter(ImageFilter.GaussianBlur(120))
img = Image.alpha_composite(img, Image.composite(glow, Image.new('RGBA', (S, S), (0, 0, 0, 0)), mask))
d = ImageDraw.Draw(img)
# RF contour rings (amber -> warm white)
for k, col, wdt in [(700, (255, 150, 60, 255), 34), (500, (255, 186, 92, 255), 38), (300, (255, 222, 150, 255), 42)]:
    d.ellipse([cx - k, cy - k * 0.82, cx + k, cy + k * 0.82], outline=col, width=wdt)
d.ellipse([cx - 120, cy - 120 * 0.82 - 10, cx + 120, cy + 120 * 0.82 - 10], fill=(255, 246, 220, 255))
img = img.resize((512, 512), Image.LANCZOS)
os.makedirs(out, exist_ok=True)
img.save(os.path.join(out, 'icon.png'))
img.resize((256, 256), Image.LANCZOS).save(os.path.join(out, 'icon_256.png'))
img.resize((64, 64), Image.LANCZOS).save(os.path.join(out, 'icon_small.png'))
img.save(os.path.join(out, 'icon.ico'), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(os.path.join(out, 'icon.icns'))
print('ok')
