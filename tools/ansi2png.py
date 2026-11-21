#!/usr/bin/env python3
"""Render an ANSI frame from `mathwave --dump` to a PNG preview (dev tool, not packaged)."""
import re, sys
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
cw = int(round(font.getlength("M"))); ch = 19
lines = open(src, encoding="utf-8").read().rstrip("\n").split("\n")
sgr = re.compile(r"\x1b\[([0-9;]*)m")
cols = max(len(sgr.sub("", l)) for l in lines)
bg = (16, 14, 22)
img = Image.new("RGB", (cols * cw + 24, len(lines) * ch + 24), bg)
d = ImageDraw.Draw(img)
for r, line in enumerate(lines):
    color, isbold, c, pos = (220, 220, 220), False, 0, 0
    for m in list(sgr.finditer(line)) + [None]:
        end = m.start() if m else len(line)
        for chr_ in line[pos:end]:
            o = ord(chr_)
            if 0x2800 < o <= 0x28FF:
                bits = o - 0x2800
                for bit, (dr, dc) in enumerate([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,0),(3,1)]):
                    if bits >> bit & 1:
                        x0 = 12 + c * cw + 1 + dc * (cw / 2); y0 = 12 + r * ch + 1 + dr * (ch / 4)
                        d.rectangle([x0, y0, x0 + cw / 2 - 2, y0 + ch / 4 - 2], fill=color)
            elif chr_ != " ":
                d.text((12 + c * cw, 12 + r * ch), chr_, fill=color, font=bold if isbold else font)
            c += 1
        if m is None:
            break
        pos = m.end()
        codes = [int(x) for x in m.group(1).split(";") if x] or [0]
        i = 0
        while i < len(codes):
            k = codes[i]
            if k == 0: color, isbold = (220, 220, 220), False
            elif k == 1: isbold = True
            elif k == 38 and codes[i + 1] == 2:
                color = tuple(codes[i + 2:i + 5]); i += 4
            i += 1
img.save(out)
print(out, img.size)
