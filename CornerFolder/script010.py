import os
import random
import shutil
import argparse

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FOLDER = "debug"

OUTPUT_RAW_MASK = "01_raw_gradient_laplacian_mask.jpg"
OUTPUT_THICK_LINE_MASK = "02_only_thick_straight_lines_mask.jpg"
OUTPUT_COLORED = "03_detected_double_line_spaces.jpg"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

GAUSSIAN_BLUR_SIZE = 3

SHARP_AMOUNT = 1.8
SHARP_BLUR_SIZE = 5

SOBEL_KERNEL = 3
LAPLACIAN_KERNEL = 3

GRADIENT_WEIGHT = 0.55
LAPLACIAN_WEIGHT = 0.45

MASK_THRESHOLD = 45

OPEN_SIZE = 1
CLOSE_SIZE = 2
DILATE_SIZE = 1

MIN_COMPONENT_AREA = 250

PRE_CLOSE_SIZE = 7
PRE_OPEN_SIZE = 3

HORIZONTAL_KERNEL_RATIO = 0.06
VERTICAL_KERNEL_RATIO = 0.06

MIN_HORIZONTAL_KERNEL = 45
MIN_VERTICAL_KERNEL = 45

LINE_THICKEN_SIZE = 5
FINAL_CLOSE_SIZE = 9

MIN_LINE_SUPPORT_RATIO = 0.35

MIN_GAP = 3
MAX_GAP = 35

TOP_BOTTOM_SEARCH_RATIO = 0.35
LEFT_RIGHT_SEARCH_RATIO = 0.35

COLOR_TOP = (0, 0, 255)
COLOR_BOTTOM = (255, 0, 0)
COLOR_LEFT = (0, 255, 0)
COLOR_RIGHT = (0, 255, 255)

FILL_ALPHA = 0.45

INNER_CORNER_OFFSET = 2

INNER_CORNER_POINT_COLOR = (0, 0, 255)
INNER_CORNER_POINT_RADIUS = 12
INNER_CORNER_RING_COLOR = (255, 255, 255)
INNER_CORNER_RING_RADIUS = 18


# ============================================================
# HELPERS
# ============================================================

def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def get_random_image_from_folder(folder):
    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not images:
        raise Exception("No images found")

    return random.choice(images)


def normalize_uint8(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def odd_kernel(size):
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


# ============================================================
# EDGE MASK
# ============================================================

def sharpen_gray(gray):
    blur = cv2.GaussianBlur(gray, (odd_kernel(SHARP_BLUR_SIZE),) * 2, 0)
    return cv2.addWeighted(gray, 1 + SHARP_AMOUNT, blur, -SHARP_AMOUNT, 0)


def make_gradient_laplacian_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (odd_kernel(GAUSSIAN_BLUR_SIZE),) * 2, 0)
    gray = sharpen_gray(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL)

    gradient = cv2.magnitude(gx, gy)
    gradient = normalize_uint8(gradient)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=LAPLACIAN_KERNEL)
    lap = normalize_uint8(np.abs(lap))

    combined = cv2.addWeighted(gradient, GRADIENT_WEIGHT, lap, LAPLACIAN_WEIGHT, 0)
    combined = normalize_uint8(combined)

    _, mask = cv2.threshold(combined, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    if OPEN_SIZE:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_SIZE, OPEN_SIZE))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if CLOSE_SIZE:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_SIZE, CLOSE_SIZE))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if DILATE_SIZE:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_SIZE, DILATE_SIZE))
        mask = cv2.dilate(mask, k, 1)

    return mask


# ============================================================
# STRONG LINE FILTER
# ============================================================

def remove_small_components(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            out[labels == i] = 255

    return out


def keep_only_thick_straight_lines(mask):
    h, w = mask.shape

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (PRE_CLOSE_SIZE, PRE_CLOSE_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, 2)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (PRE_OPEN_SIZE, PRE_OPEN_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, 1)

    mask = remove_small_components(mask)

    hk = max(MIN_HORIZONTAL_KERNEL, int(w * HORIZONTAL_KERNEL_RATIO))
    vk = max(MIN_VERTICAL_KERNEL, int(h * VERTICAL_KERNEL_RATIO))

    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 3)))

    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, vk)))

    straight = cv2.bitwise_or(horizontal, vertical)

    straight = cv2.dilate(
        straight,
        cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_THICKEN_SIZE, LINE_THICKEN_SIZE)),
        1
    )

    straight = cv2.morphologyEx(
        straight,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (FINAL_CLOSE_SIZE, FINAL_CLOSE_SIZE)),
        2
    )

    return remove_small_components(straight)


# ============================================================
# DOUBLE LINE DETECTION
# ============================================================

def find_double_horizontal(mask, y0, y1):
    roi = mask[y0:y1]
    support = np.count_nonzero(roi, axis=1)

    thresh = int(mask.shape[1] * MIN_LINE_SUPPORT_RATIO)
    rows = np.where(support >= thresh)[0]

    best = None
    best_score = -1

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            gap = b - a

            if gap < MIN_GAP:
                continue
            if gap > MAX_GAP:
                break

            mid = roi[a:b]
            empty = 1 - np.count_nonzero(mid) / mid.size

            score = empty
            if score > best_score:
                best_score = score
                best = (y0 + a, y0 + b, score)

    return best


def find_double_vertical(mask, x0, x1):
    roi = mask[:, x0:x1]
    support = np.count_nonzero(roi, axis=0)

    thresh = int(mask.shape[0] * MIN_LINE_SUPPORT_RATIO)
    cols = np.where(support >= thresh)[0]

    best = None
    best_score = -1

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            gap = b - a

            if gap < MIN_GAP:
                continue
            if gap > MAX_GAP:
                break

            mid = roi[:, a:b]
            empty = 1 - np.count_nonzero(mid) / mid.size

            score = empty
            if score > best_score:
                best_score = score
                best = (x0 + a, x0 + b, score)

    return best


# ============================================================
# VISUALIZATION
# ============================================================

def draw(image, mask):
    h, w = mask.shape
    out = image.copy()
    overlay = image.copy()

    top = find_double_horizontal(mask, 0, int(h * TOP_BOTTOM_SEARCH_RATIO))
    bottom = find_double_horizontal(mask, int(h * (1 - TOP_BOTTOM_SEARCH_RATIO)), h)

    left = find_double_vertical(mask, 0, int(w * LEFT_RIGHT_SEARCH_RATIO))
    right = find_double_vertical(mask, int(w * (1 - LEFT_RIGHT_SEARCH_RATIO)), w)

    for r, color, name in [
        (top, COLOR_TOP, "top"),
        (bottom, COLOR_BOTTOM, "bottom"),
    ]:
        if r:
            y1, y2, _ = r
            cv2.rectangle(overlay, (0, y1), (w, y2), color, -1)

    for r, color, name in [
        (left, COLOR_LEFT, "left"),
        (right, COLOR_RIGHT, "right"),
    ]:
        if r:
            x1, x2, _ = r
            cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)

    return cv2.addWeighted(overlay, FILL_ALPHA, out, 1 - FILL_ALPHA, 0)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()

    clean_folder(OUTPUT_FOLDER)

    img_path = get_random_image_from_folder(args.folder)
    img = cv2.imread(img_path)

    raw = make_gradient_laplacian_mask(img)
    thick = keep_only_thick_straight_lines(raw)

    result = draw(img, thick)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, OUTPUT_RAW_MASK), raw)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, OUTPUT_THICK_LINE_MASK), thick)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, OUTPUT_COLORED), result)

    print("Done:", img_path)


if __name__ == "__main__":
    main()