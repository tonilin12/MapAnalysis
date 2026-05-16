from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

INPUT_FOLDER = "test_maps"
OUTPUT_FOLDER_NAME = "top_line_results"

MAX_SCREEN_WIDTH = 1200
MAX_SCREEN_HEIGHT = 800

MIN_RECT_AREA_RATIO = 0.20
EPSILON_RATIO = 0.02

MAX_PEELS = 4
PAD_INSIDE = 12

CROP_PADDING_X = 5

EXPAND_LINE_RATIO = 0.08

MAX_ABOVE_HEIGHT = None
# MAX_ABOVE_HEIGHT = 250

LINE_SEARCH_RADIUS = 35
LINE_THICKNESS_RADIUS = 4

KEEP_LINE_THICKNESS_RADIUS = 4

DARK_THRESHOLD = 170

MIN_HORIZONTAL_WIDTH_RATIO = 0.45


# ============================================================
# IO HELPERS
# ============================================================

def get_script_dir():
    return Path(__file__).resolve().parent


def get_input_folder():
    return get_script_dir() / INPUT_FOLDER


def get_output_folder():
    output_folder = get_script_dir() / OUTPUT_FOLDER_NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def get_first_image(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    images.sort()

    if len(images) == 0:
        raise RuntimeError("No image files found in folder.")

    return images[0]


def resize_to_fit_screen(img, max_width=MAX_SCREEN_WIDTH, max_height=MAX_SCREEN_HEIGHT):
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blur,
        60,
        180
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    edges = cv2.dilate(edges, kernel, iterations=1)

    return gray, edges


# ============================================================
# RECTANGLE DETECTION
# ============================================================

def find_inner_rectangle(img):
    h, w = img.shape[:2]

    gray, edges = preprocess(img)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < w * h * MIN_RECT_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            EPSILON_RATIO * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        x, y, rw, rh = cv2.boundingRect(approx)

        rect_area = rw * rh
        fill_ratio = area / max(rect_area, 1)

        if fill_ratio < 0.70:
            continue

        candidates.append({
            "contour": approx,
            "area": area,
            "x": x,
            "y": y,
            "w": rw,
            "h": rh,
        })

    if len(candidates) == 0:
        return None, edges

    best = min(
        candidates,
        key=lambda c: c["area"]
    )

    return best, edges


def get_top_horizontal_line(rect):
    points = rect["contour"].reshape(-1, 2)

    points_sorted = sorted(points, key=lambda p: p[1])

    top_two = points_sorted[:2]
    top_two = sorted(top_two, key=lambda p: p[0])

    left_point = top_two[0]
    right_point = top_two[1]

    left_x = int(left_point[0])
    right_x = int(right_point[0])

    top_y = int(round((left_point[1] + right_point[1]) / 2))

    return left_x, top_y, right_x, top_y


# ============================================================
# PEELING
# ============================================================

def compute_next_crop_from_rect(rect, current_img):
    h, w = current_img.shape[:2]

    x1 = max(rect["x"] + PAD_INSIDE, 0)
    y1 = max(rect["y"] + PAD_INSIDE, 0)
    x2 = min(rect["x"] + rect["w"] - PAD_INSIDE, w)
    y2 = min(rect["y"] + rect["h"] - PAD_INSIDE, h)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def collect_peel_top_lines(original):
    peel_results = []

    current = original.copy()
    offset_x = 0
    offset_y = 0

    last_edges = None

    for peel in range(MAX_PEELS):
        rect, edges = find_inner_rectangle(current)
        last_edges = edges

        if rect is None:
            break

        local_line = get_top_horizontal_line(rect)

        gx1 = offset_x + local_line[0]
        gy1 = offset_y + local_line[1]
        gx2 = offset_x + local_line[2]
        gy2 = offset_y + local_line[3]

        global_rect = {
            "contour": rect["contour"] + np.array([[[offset_x, offset_y]]]),
            "x": offset_x + rect["x"],
            "y": offset_y + rect["y"],
            "w": rect["w"],
            "h": rect["h"],
            "area": rect["area"],
            "peel": peel,
        }

        peel_results.append({
            "peel": peel,
            "rect": global_rect,
            "top_line": (gx1, gy1, gx2, gy2),
        })

        next_crop = compute_next_crop_from_rect(rect, current)

        if next_crop is None:
            break

        nx1, ny1, nx2, ny2 = next_crop

        current = current[ny1:ny2, nx1:nx2]

        offset_x += nx1
        offset_y += ny1

    if len(peel_results) == 0:
        raise RuntimeError("No rectangle found in any peel.")

    return peel_results, last_edges


# ============================================================
# STRONGEST DARK / THICK HORIZONTAL LINE
# ============================================================

def score_horizontal_band(gray, x1, x2, y):
    h, w = gray.shape[:2]

    y1 = max(0, y - LINE_THICKNESS_RADIUS)
    y2 = min(h, y + LINE_THICKNESS_RADIUS + 1)

    x1 = max(0, x1)
    x2 = min(w, x2)

    if x2 <= x1 or y2 <= y1:
        return -1

    band = gray[y1:y2, x1:x2]

    darkness = 255.0 - band.astype(np.float32)
    dark_pixels = band < DARK_THRESHOLD

    dark_ratio = np.mean(dark_pixels)
    mean_darkness = np.mean(darkness)

    row_dark_counts = np.sum(dark_pixels, axis=1)
    thickness_score = np.max(row_dark_counts) / max(1, x2 - x1)

    score = (
        mean_darkness
        + 120.0 * dark_ratio
        + 120.0 * thickness_score
    )

    return score


def fit_strongest_top_horizontal_line(original, peel_results):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    best = None

    for result in peel_results:
        x1, y1, x2, y2 = result["top_line"]

        left_x = min(x1, x2)
        right_x = max(x1, x2)

        line_len = right_x - left_x

        if line_len < original.shape[1] * MIN_HORIZONTAL_WIDTH_RATIO:
            continue

        search_y1 = max(0, y1 - LINE_SEARCH_RADIUS)
        search_y2 = min(original.shape[0] - 1, y1 + LINE_SEARCH_RADIUS)

        for y in range(search_y1, search_y2 + 1):
            score = score_horizontal_band(
                gray,
                left_x,
                right_x,
                y
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "peel": result["peel"],
                    "x1": left_x,
                    "y1": y,
                    "x2": right_x,
                    "y2": y,
                }

    if best is None:
        raise RuntimeError("Could not fit strongest horizontal top line.")

    return best


def expand_top_line(line, img_width):
    x1 = int(line["x1"])
    y1 = int(line["y1"])
    x2 = int(line["x2"])
    y2 = int(line["y2"])

    left_x = min(x1, x2)
    right_x = max(x1, x2)

    line_len = right_x - left_x
    expand_amount = int(line_len * EXPAND_LINE_RATIO)

    expanded_x1 = max(0, left_x - expand_amount)
    expanded_x2 = min(img_width - 1, right_x + expand_amount)

    return {
        "x1": expanded_x1,
        "y1": y1,
        "x2": expanded_x2,
        "y2": y2,
        "score": line["score"],
        "peel": line["peel"],
    }


# ============================================================
# CROP ABOVE LINE + LINE ITSELF ONLY
# ============================================================

def crop_above_line_plus_line_itself(img, line):
    h, w = img.shape[:2]

    top_y = int(round((line["y1"] + line["y2"]) / 2))

    left_x = min(line["x1"], line["x2"])
    right_x = max(line["x1"], line["x2"])

    crop_y2 = min(h, top_y + KEEP_LINE_THICKNESS_RADIUS + 1)

    if MAX_ABOVE_HEIGHT is None:
        crop_y1 = 0
    else:
        crop_y1 = max(0, crop_y2 - MAX_ABOVE_HEIGHT)

    crop_x1 = max(0, left_x - CROP_PADDING_X)
    crop_x2 = min(w, right_x + CROP_PADDING_X)

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped, {
        "x1": crop_x1,
        "y1": crop_y1,
        "x2": crop_x2,
        "y2": crop_y2,
    }


# ============================================================
# DRAWING
# ============================================================

def draw_result(img, peel_results, strongest_line, expanded_line, crop_box):
    debug = img.copy()

    for result in peel_results:
        rect = result["rect"]
        line = result["top_line"]

        cv2.drawContours(
            debug,
            [rect["contour"]],
            -1,
            (100, 100, 100),
            2
        )

        cv2.line(
            debug,
            (line[0], line[1]),
            (line[2], line[3]),
            (180, 180, 180),
            2
        )

    cv2.line(
        debug,
        (strongest_line["x1"], strongest_line["y1"]),
        (strongest_line["x2"], strongest_line["y2"]),
        (0, 0, 255),
        5
    )

    cv2.line(
        debug,
        (expanded_line["x1"], expanded_line["y1"]),
        (expanded_line["x2"], expanded_line["y2"]),
        (255, 0, 255),
        3
    )

    cv2.rectangle(
        debug,
        (crop_box["x1"], crop_box["y1"]),
        (crop_box["x2"], crop_box["y2"]),
        (255, 0, 0),
        4
    )

    cv2.putText(
        debug,
        "KEEP ABOVE LINE + LINE ITSELF ONLY",
        (crop_box["x1"], max(40, crop_box["y2"] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    return debug


# ============================================================
# HIGH-LEVEL PIPELINE
# ============================================================

def process_image(img):
    h, w = img.shape[:2]

    peel_results, last_edges = collect_peel_top_lines(img)

    strongest_line = fit_strongest_top_horizontal_line(
        img,
        peel_results
    )

    expanded_line = expand_top_line(
        strongest_line,
        w
    )

    cropped, crop_box = crop_above_line_plus_line_itself(
        img,
        expanded_line
    )

    debug = draw_result(
        img,
        peel_results,
        strongest_line,
        expanded_line,
        crop_box
    )

    return {
        "cropped": cropped,
        "debug": debug,
        "last_edges": last_edges,
        "peel_results": peel_results,
        "strongest_line": strongest_line,
        "expanded_line": expanded_line,
        "crop_box": crop_box,
    }


def save_outputs(result):
    output_folder = get_output_folder()

    debug_path = output_folder / "debug_strongest_top_line_area.png"
    cropped_path = output_folder / "debug_area_above_line_plus_line_itself.png"
    edges_path = output_folder / "debug_last_edges.png"

    cv2.imwrite(str(debug_path), result["debug"])
    cv2.imwrite(str(cropped_path), result["cropped"])

    if result["last_edges"] is not None:
        cv2.imwrite(str(edges_path), result["last_edges"])

    print("\nSaved results to:")
    print(output_folder)


def print_report(result):
    print("\nPeel top-line candidates:")

    for item in result["peel_results"]:
        line = item["top_line"]
        print(
            f"peel={item['peel']} "
            f"x1={line[0]} y1={line[1]} "
            f"x2={line[2]} y2={line[3]}"
        )

    strongest_line = result["strongest_line"]
    crop_box = result["crop_box"]

    print("\nStrongest fitted horizontal top line:")
    print(f"from peel = {strongest_line['peel']}")
    print(f"score = {round(strongest_line['score'], 2)}")
    print(f"x1 = {strongest_line['x1']}")
    print(f"y1 = {strongest_line['y1']}")
    print(f"x2 = {strongest_line['x2']}")
    print(f"y2 = {strongest_line['y2']}")

    print("\nKept crop area:")
    print(f"x1 = {crop_box['x1']}")
    print(f"y1 = {crop_box['y1']}")
    print(f"x2 = {crop_box['x2']}")
    print(f"y2 = {crop_box['y2']}")

    print("\nLine kept radius:")
    print(f"KEEP_LINE_THICKNESS_RADIUS = {KEEP_LINE_THICKNESS_RADIUS}")


def show_result(result):
    display_result, _ = resize_to_fit_screen(result["debug"])
    display_crop, _ = resize_to_fit_screen(result["cropped"])

    cv2.imshow("Strongest top line and kept area", display_result)
    cv2.imshow("Above line plus line itself only", display_crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    input_folder = get_input_folder()

    image_path = get_first_image(input_folder)

    print("Input image:")
    print(image_path)

    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    result = process_image(img)

    save_outputs(result)

    print_report(result)

    show_result(result)


if __name__ == "__main__":
    main()