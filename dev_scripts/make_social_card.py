"""Generate the mbirtorch social-preview card (Open Graph / GitHub).

The card is 1280x640, the size GitHub and the Open Graph protocol both
accept, so one image serves the GitHub repository preview and the
documentation link preview.  The MBIRTorch wordmark sits over a
sinogram: each curve is one object's projection, s(theta) = A
sin(theta + phi) across projection angle theta in [0, pi], which is
what a sinogram of that object is.  The card renders at twice the final
size and downsamples once, which anti-aliases the text and the curves.

The output is written to docs/source/_static/mbirtorch_card.png, the
location docs/source/conf.py points the og:image meta tag at.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SCALE = 2
W, H = 1280 * SCALE, 640 * SCALE

TORCH_RED = (238, 42, 74)
LIGHT = (238, 238, 240)
MUTED = (150, 152, 160)
BG_TOP = (17, 18, 22)
BG_BOTTOM = (9, 9, 12)

WORD_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 168 * SCALE)
TAG_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial.ttf", 44 * SCALE)
URL_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32 * SCALE)


def background():
    """Returns the dark vertical-gradient background."""
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        f = y / (H - 1)
        c = tuple(int(BG_TOP[k] + f * (BG_BOTTOM[k] - BG_TOP[k]))
                  for k in range(3))
        for x in range(W):
            px[x, y] = c
    return img


# Objects as (amplitude 0..1 of half band, phase, brightness, thickness).
OBJECTS = [
    (0.85, 0.3, 1.00, 5), (0.60, 1.7, 0.85, 4), (0.40, 3.0, 0.90, 4),
    (0.72, 2.3, 0.70, 3), (0.25, 0.9, 0.95, 6), (0.52, 4.2, 0.75, 3),
    (0.33, 5.0, 0.80, 4), (0.90, 5.6, 0.60, 3), (0.15, 2.0, 1.00, 7),
]


def draw_sinogram(img):
    """Draws the sinogram as glowing sinusoidal threads across the
    lower band, with a blurred copy behind for the glow."""
    left, right = 60 * SCALE, W - 60 * SCALE
    band_cy = 500 * SCALE
    half = 95 * SCALE
    thetas = np.linspace(0, np.pi, 600)
    xs = left + thetas / np.pi * (right - left)

    threads = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(threads)
    for amp, phase, bright, thick in OBJECTS:
        ys = band_cy - amp * half * np.sin(thetas + phase)
        pts = list(zip(xs.tolist(), ys.tolist()))
        c = (255, int(70 + 60 * bright), int(90 + 50 * bright),
             int(230 * bright))
        d.line(pts, fill=c, width=thick * SCALE, joint="curve")

    glow = threads.filter(ImageFilter.GaussianBlur(6 * SCALE))
    img.paste(glow, (0, 0), glow)
    img.paste(threads, (0, 0), threads)


def draw_wordmark(img):
    """Draws 'MBIRTorch' centered: MBIR light, Torch in the house red,
    with a soft red glow behind the word."""
    pieces = [("MBIR", LIGHT), ("Torch", TORCH_RED)]
    d = ImageDraw.Draw(img)
    widths = [d.textbbox((0, 0), t, font=WORD_FONT)[2] for t, _ in pieces]
    total = sum(widths)
    x = (W - total) // 2
    y = 92 * SCALE

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((x, y), "MBIRTorch", font=WORD_FONT,
                              fill=TORCH_RED + (90,))
    glow = glow.filter(ImageFilter.GaussianBlur(16 * SCALE))
    img.paste(glow, (0, 0), glow)

    for (t, color), w in zip(pieces, widths):
        d.text((x, y), t, font=WORD_FONT, fill=color)
        x += w


def draw_text(img):
    """Draws the tagline and the documentation URL, both centered."""
    d = ImageDraw.Draw(img)
    tag = "Model-based iterative tomographic reconstruction in PyTorch"
    tw = d.textbbox((0, 0), tag, font=TAG_FONT)[2]
    d.text(((W - tw) // 2, 292 * SCALE), tag, font=TAG_FONT, fill=MUTED)

    url = "mbirtorch.readthedocs.io"
    uw = d.textbbox((0, 0), url, font=URL_FONT)[2]
    d.text(((W - uw) // 2, 356 * SCALE), url, font=URL_FONT, fill=TORCH_RED)


def main():
    img = background().convert("RGBA")
    draw_sinogram(img)
    draw_wordmark(img)
    draw_text(img)
    img = img.convert("RGB").resize((W // SCALE, H // SCALE), Image.LANCZOS)
    out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "..", "docs", "source", "_static",
                       "mbirtorch_card.png")
    img.save(os.path.normpath(out))
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
