import math
import cv2
import numpy as np


CORNER_ORDER = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]

OUTPUT_RAW_MASK = "01_raw_gradient_laplacian_mask.jpg"
OUTPUT_THICK_LINE_MASK = "02_only_thick_straight_lines_mask.jpg"
OUTPUT_FINAL = "03_final_corners.jpg"


# ============================================================
# DETECTION CONFIG
# ============================================================

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

INNER_CORNER_OFFSET = 2


# ============================================================
# REFINEMENT CONFIG
# ============================================================

CROSS_SEARCH_RADIUS = 16
CORNER_NEIGHBORHOOD_RADIUS = 60

REFINE_DARK_THRESHOLD = 175

H_REFINE_KERNEL_SIZE = 25
V_REFINE_KERNEL_SIZE = 25

MIN_REFINE_LINE_SUPPORT = 10
REFINE_DISTANCE_PENALTY = 1.5


# ============================================================
# DRAWING CONFIG
# ============================================================

# OpenCV uses BGR.
# Dark green = final refined rectangle.
# Dark blue = before-refine rectangle.
COLOR_REFINED = (0, 120, 0)
COLOR_BEFORE_REFINE = (160, 70, 0)
COLOR_TEXT = (255, 255, 255)


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def odd_kernel(size):
    size = int(size)

    if size < 1:
        size = 1

    if size % 2 == 0:
        size += 1

    return size


def normalize_uint8(image):
    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def make_rect_kernel(w, h):
    return cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, int(w)), max(1, int(h))),
    )


# ============================================================
# MASK BUILDING
# ============================================================

def sharpen_gray(gray):
    blur_size = odd_kernel(SHARP_BLUR_SIZE)

    blurred = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        0,
    )

    sharp = cv2.addWeighted(
        gray,
        1.0 + SHARP_AMOUNT,
        blurred,
        -SHARP_AMOUNT,
        0,
    )

    return sharp


def make_gradient_laplacian_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (
            odd_kernel(GAUSSIAN_BLUR_SIZE),
            odd_kernel(GAUSSIAN_BLUR_SIZE),
        ),
        0,
    )

    gray = sharpen_gray(gray)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=SOBEL_KERNEL,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=SOBEL_KERNEL,
    )

    gradient = cv2.magnitude(gx, gy)
    gradient = normalize_uint8(gradient)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=LAPLACIAN_KERNEL,
    )

    laplacian = np.abs(laplacian)
    laplacian = normalize_uint8(laplacian)

    combined = cv2.addWeighted(
        gradient,
        GRADIENT_WEIGHT,
        laplacian,
        LAPLACIAN_WEIGHT,
        0,
    )

    combined = normalize_uint8(combined)

    _, mask = cv2.threshold(
        combined,
        MASK_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    if OPEN_SIZE > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            make_rect_kernel(OPEN_SIZE, OPEN_SIZE),
        )

    if CLOSE_SIZE > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            make_rect_kernel(CLOSE_SIZE, CLOSE_SIZE),
        )

    if DILATE_SIZE > 0:
        mask = cv2.dilate(
            mask,
            make_rect_kernel(DILATE_SIZE, DILATE_SIZE),
            iterations=1,
        )

    return mask


def remove_small_components(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def keep_only_thick_straight_lines(mask):
    h, w = mask.shape[:2]

    cleaned = mask.copy()

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        make_rect_kernel(PRE_CLOSE_SIZE, PRE_CLOSE_SIZE),
        iterations=2,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        make_rect_kernel(PRE_OPEN_SIZE, PRE_OPEN_SIZE),
        iterations=1,
    )

    cleaned = remove_small_components(
        cleaned,
        MIN_COMPONENT_AREA,
    )

    horizontal_kernel_w = max(
        MIN_HORIZONTAL_KERNEL,
        int(w * HORIZONTAL_KERNEL_RATIO),
    )

    vertical_kernel_h = max(
        MIN_VERTICAL_KERNEL,
        int(h * VERTICAL_KERNEL_RATIO),
    )

    horizontal = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        make_rect_kernel(horizontal_kernel_w, 3),
        iterations=1,
    )

    vertical = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        make_rect_kernel(3, vertical_kernel_h),
        iterations=1,
    )

    straight = cv2.bitwise_or(horizontal, vertical)

    straight = cv2.dilate(
        straight,
        make_rect_kernel(LINE_THICKEN_SIZE, LINE_THICKEN_SIZE),
        iterations=1,
    )

    straight = cv2.morphologyEx(
        straight,
        cv2.MORPH_CLOSE,
        make_rect_kernel(FINAL_CLOSE_SIZE, FINAL_CLOSE_SIZE),
        iterations=2,
    )

    straight = remove_small_components(
        straight,
        MIN_COMPONENT_AREA,
    )

    return straight


# ============================================================
# CORRIDOR DETECTION
# ============================================================

def find_double_horizontal_space(mask, y_start, y_end):
    h, w = mask.shape[:2]

    roi = mask[y_start:y_end, :]

    if roi.size == 0:
        return None

    row_support = np.count_nonzero(roi > 0, axis=1)
    support_threshold = int(w * MIN_LINE_SUPPORT_RATIO)

    strong_rows = np.where(row_support >= support_threshold)[0]

    best = None
    best_score = -1

    for i in range(len(strong_rows)):
        for j in range(i + 1, len(strong_rows)):
            y1 = strong_rows[i]
            y2 = strong_rows[j]

            gap = y2 - y1

            if gap < MIN_GAP:
                continue

            if gap > MAX_GAP:
                break

            middle = roi[y1 + 1:y2, :]

            if middle.size == 0:
                continue

            empty_score = 1.0 - (
                np.count_nonzero(middle > 0) / middle.size
            )

            line_score = (
                row_support[y1] + row_support[y2]
            ) / (2.0 * w)

            score = empty_score * 0.65 + line_score * 0.35

            if score > best_score:
                best_score = score
                best = (
                    y_start + y1,
                    y_start + y2,
                    best_score,
                )

    return best


def find_double_vertical_space(mask, x_start, x_end):
    h, w = mask.shape[:2]

    roi = mask[:, x_start:x_end]

    if roi.size == 0:
        return None

    col_support = np.count_nonzero(roi > 0, axis=0)
    support_threshold = int(h * MIN_LINE_SUPPORT_RATIO)

    strong_cols = np.where(col_support >= support_threshold)[0]

    best = None
    best_score = -1

    for i in range(len(strong_cols)):
        for j in range(i + 1, len(strong_cols)):
            x1 = strong_cols[i]
            x2 = strong_cols[j]

            gap = x2 - x1

            if gap < MIN_GAP:
                continue

            if gap > MAX_GAP:
                break

            middle = roi[:, x1 + 1:x2]

            if middle.size == 0:
                continue

            empty_score = 1.0 - (
                np.count_nonzero(middle > 0) / middle.size
            )

            line_score = (
                col_support[x1] + col_support[x2]
            ) / (2.0 * h)

            score = empty_score * 0.65 + line_score * 0.35

            if score > best_score:
                best_score = score
                best = (
                    x_start + x1,
                    x_start + x2,
                    best_score,
                )

    return best


def get_inner_corridor_intersection_points(top, bottom, left, right, width, height):
    if top is None or bottom is None or left is None or right is None:
        return None

    _, top_inner_y, _ = top
    bottom_inner_y, _, _ = bottom

    _, left_inner_x, _ = left
    right_inner_x, _, _ = right

    offset = INNER_CORNER_OFFSET

    points = {
        "top_left": (
            left_inner_x + offset,
            top_inner_y + offset,
        ),
        "top_right": (
            right_inner_x - offset,
            top_inner_y + offset,
        ),
        "bottom_right": (
            right_inner_x - offset,
            bottom_inner_y - offset,
        ),
        "bottom_left": (
            left_inner_x + offset,
            bottom_inner_y - offset,
        ),
    }

    safe_points = {}

    for name, (x, y) in points.items():
        safe_points[name] = (
            clamp(int(round(x)), 0, width - 1),
            clamp(int(round(y)), 0, height - 1),
        )

    return safe_points


def detect_initial_corners(image_bgr):
    h, w = image_bgr.shape[:2]

    raw_mask = make_gradient_laplacian_mask(image_bgr)
    thick_line_mask = keep_only_thick_straight_lines(raw_mask)

    top_end = int(h * TOP_BOTTOM_SEARCH_RATIO)
    bottom_start = int(h * (1.0 - TOP_BOTTOM_SEARCH_RATIO))

    left_end = int(w * LEFT_RIGHT_SEARCH_RATIO)
    right_start = int(w * (1.0 - LEFT_RIGHT_SEARCH_RATIO))

    top = find_double_horizontal_space(thick_line_mask, 0, top_end)
    bottom = find_double_horizontal_space(thick_line_mask, bottom_start, h)
    left = find_double_vertical_space(thick_line_mask, 0, left_end)
    right = find_double_vertical_space(thick_line_mask, right_start, w)

    points = get_inner_corridor_intersection_points(
        top,
        bottom,
        left,
        right,
        w,
        h,
    )

    debug_info = {
        "raw_mask": raw_mask,
        "thick_line_mask": thick_line_mask,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }

    return points, debug_info


def detect_corners_with_second_script_logic(image_bgr):
    return detect_initial_corners(image_bgr)


# ============================================================
# CORNER REFINEMENT
# ============================================================

def crop_neighborhood(img, center, radius):
    h, w = img.shape[:2]

    cx, cy = center

    x1 = clamp(cx - radius, 0, w - 1)
    y1 = clamp(cy - radius, 0, h - 1)
    x2 = clamp(cx + radius, 0, w - 1)
    y2 = clamp(cy + radius, 0, h - 1)

    roi = img[y1:y2 + 1, x1:x2 + 1].copy()

    return roi, x1, y1


def build_refine_dark_mask(gray):
    _, dark = cv2.threshold(
        gray,
        REFINE_DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )

    return dark


def detect_refine_line_masks(dark):
    h_mask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        make_rect_kernel(H_REFINE_KERNEL_SIZE, 1),
    )

    v_mask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        make_rect_kernel(1, V_REFINE_KERNEL_SIZE),
    )

    h_mask = cv2.dilate(
        h_mask,
        make_rect_kernel(5, 1),
        iterations=1,
    )

    v_mask = cv2.dilate(
        v_mask,
        make_rect_kernel(1, 5),
        iterations=1,
    )

    return h_mask, v_mask


def score_cross_candidate(h_mask, v_mask, x, y):
    h, w = h_mask.shape[:2]

    left = max(0, x - CROSS_SEARCH_RADIUS)
    right = min(w - 1, x + CROSS_SEARCH_RADIUS)
    top = max(0, y - CROSS_SEARCH_RADIUS)
    bottom = min(h - 1, y + CROSS_SEARCH_RADIUS)

    h_row = h_mask[y, left:right + 1]
    v_col = v_mask[top:bottom + 1, x]

    h_support = int(np.count_nonzero(h_row))
    v_support = int(np.count_nonzero(v_col))

    direct_hit = 0

    if h_mask[y, x] > 0:
        direct_hit += 1

    if v_mask[y, x] > 0:
        direct_hit += 1

    score = h_support + v_support + direct_hit * 25

    if h_support < MIN_REFINE_LINE_SUPPORT:
        score -= 1000

    if v_support < MIN_REFINE_LINE_SUPPORT:
        score -= 1000

    return score, h_support, v_support


def refine_corner_by_cross(original, predicted_corner):
    roi, rx1, ry1 = crop_neighborhood(
        original,
        predicted_corner,
        CORNER_NEIGHBORHOOD_RADIUS,
    )

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    dark = build_refine_dark_mask(gray)
    h_mask, v_mask = detect_refine_line_masks(dark)

    local_guess_x = int(round(predicted_corner[0] - rx1))
    local_guess_y = int(round(predicted_corner[1] - ry1))

    h, w = gray.shape[:2]

    sx1 = clamp(local_guess_x - CROSS_SEARCH_RADIUS, 0, w - 1)
    sy1 = clamp(local_guess_y - CROSS_SEARCH_RADIUS, 0, h - 1)
    sx2 = clamp(local_guess_x + CROSS_SEARCH_RADIUS, 0, w - 1)
    sy2 = clamp(local_guess_y + CROSS_SEARCH_RADIUS, 0, h - 1)

    best = None

    for y in range(sy1, sy2 + 1):
        for x in range(sx1, sx2 + 1):
            score, h_support, v_support = score_cross_candidate(
                h_mask,
                v_mask,
                x,
                y,
            )

            dist = math.sqrt(
                (x - local_guess_x) ** 2 +
                (y - local_guess_y) ** 2,
            )

            final_score = score - dist * REFINE_DISTANCE_PENALTY

            if best is None or final_score > best["score"]:
                best = {
                    "score": final_score,
                    "point": (rx1 + x, ry1 + y),
                    "h_support": h_support,
                    "v_support": v_support,
                }

    if best is None:
        return predicted_corner, {
            "score": None,
            "h_support": 0,
            "v_support": 0,
        }

    return best["point"], best


def refine_all_corners_by_cross(original, predicted_points):
    refined = {}
    refine_info = {}

    for corner in CORNER_ORDER:
        refined_point, info = refine_corner_by_cross(
            original,
            predicted_points[corner],
        )

        refined[corner] = refined_point
        refine_info[corner] = info

    return refined, refine_info


def detect_and_refine_corners(image_bgr):
    predicted_points, debug_info = detect_initial_corners(image_bgr)

    if predicted_points is None:
        return None, None

    refined_points, refine_info = refine_all_corners_by_cross(
        image_bgr,
        predicted_points,
    )

    debug_info["before_refine"] = predicted_points
    debug_info["after_refine"] = refined_points
    debug_info["refine_info"] = refine_info

    return refined_points, debug_info


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_point(img, point, color, label=None, radius=8):
    x, y = point

    cv2.circle(
        img,
        (int(x), int(y)),
        radius,
        color,
        -1,
    )

    if label is not None:
        cv2.putText(
            img,
            label,
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )


def draw_polygon(img, points, color, thickness=4):
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]

        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color,
            thickness,
        )


def draw_corner_set(img, points_dict, color, label_prefix, thickness=4, radius=8):
    points = [points_dict[name] for name in CORNER_ORDER]

    draw_polygon(img, points, color, thickness)

    for name in CORNER_ORDER:
        draw_point(
            img,
            points_dict[name],
            color,
            f"{label_prefix}_{name}",
            radius=radius,
        )


def draw_final_corners(original, predicted_points, before_refine=None):
    debug = original.copy()

    if before_refine is not None:
        draw_corner_set(
            debug,
            before_refine,
            COLOR_BEFORE_REFINE,
            "before",
            thickness=2,
            radius=5,
        )

    draw_corner_set(
        debug,
        predicted_points,
        COLOR_REFINED,
        "final",
        thickness=4,
        radius=8,
    )

    cv2.putText(
        debug,
        "dark blue = before refine | dark green = refined",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )

    return debug