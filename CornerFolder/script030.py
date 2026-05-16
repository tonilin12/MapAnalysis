import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from map_detector_project.CornerFolder.corner_utils import (
    CORNER_ORDER,
    detect_and_refine_corners,
)


# ============================================================
# CONFIG
# ============================================================

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

RESULTS_FOLDER = "results"
SHARED_OUTPUT_FOLDER = "final_corner_review_images"

OUTPUT_SUFFIX = "_final_corner_review.jpg"

MAX_WORKERS = None


# ============================================================
# REVIEW IMAGE CONFIG
# ============================================================

NEIGHBOURHOOD_RADIUS = 180
SEPARATOR_SIZE = 18

COLOR_POINT = (120, 60, 20)
COLOR_POINT_RING = (80, 35, 10)
COLOR_LINE = (100, 50, 15)
COLOR_CROSS = (130, 65, 20)

COLOR_TEXT = (220, 220, 220)
COLOR_SEPARATOR = (35, 35, 35)
COLOR_LABEL_BG = (18, 18, 18)
COLOR_BACKGROUND = (10, 10, 10)


# ============================================================
# IO HELPERS
# ============================================================

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return img


def get_script_dir():
    return Path(__file__).resolve().parent


def get_input_folder():
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = get_script_dir() / "test_maps"

    folder = folder.resolve()

    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    return folder


def get_results_root():
    results_root = get_script_dir() / RESULTS_FOLDER
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def prepare_shared_output_folder(results_root):
    folder = results_root / SHARED_OUTPUT_FOLDER

    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)

    return folder


def get_image_files(folder):
    image_files = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        image_files.append(path)

    image_files.sort()
    return image_files


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def seconds_since(start_time):
    return time.perf_counter() - start_time


def choose_worker_count():
    if MAX_WORKERS is not None:
        return MAX_WORKERS

    cpu_count = os.cpu_count() or 4

    return max(1, min(cpu_count, 8))


# ============================================================
# REVIEW IMAGE HELPERS
# ============================================================

def resize_to_height(image, target_height):
    h, w = image.shape[:2]

    if h == target_height:
        return image

    scale = target_height / h
    new_w = int(round(w * scale))

    return cv2.resize(
        image,
        (new_w, target_height),
        interpolation=cv2.INTER_AREA,
    )


def pad_to_size(image, target_width, target_height):
    h, w = image.shape[:2]

    padded = np.full(
        (target_height, target_width, 3),
        COLOR_BACKGROUND,
        dtype=np.uint8,
    )

    paste_x = (target_width - w) // 2
    paste_y = (target_height - h) // 2

    padded[paste_y:paste_y + h, paste_x:paste_x + w] = image

    return padded


def make_separator(height, width):
    return np.full(
        (height, width, 3),
        COLOR_SEPARATOR,
        dtype=np.uint8,
    )


def draw_final_corners_on_original(image, final_corners):
    out = image.copy()

    ordered_points = [
        final_corners["top_left"],
        final_corners["top_right"],
        final_corners["bottom_right"],
        final_corners["bottom_left"],
    ]

    for i in range(len(ordered_points)):
        p1 = ordered_points[i]
        p2 = ordered_points[(i + 1) % len(ordered_points)]

        cv2.line(
            out,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            COLOR_LINE,
            5,
        )

    for name in CORNER_ORDER:
        x, y = final_corners[name]

        x = int(round(x))
        y = int(round(y))

        cv2.circle(out, (x, y), 10, COLOR_POINT, -1)
        cv2.circle(out, (x, y), 16, COLOR_POINT_RING, 2)

        cv2.putText(
            out,
            name,
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLOR_TEXT,
            3,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        out,
        (0, 0),
        (430, 60),
        COLOR_LABEL_BG,
        -1,
    )

    cv2.putText(
        out,
        "final corners on original image",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )

    return out


def crop_corner_neighbourhood(image, point, radius):
    h, w = image.shape[:2]

    x, y = point
    x = int(round(x))
    y = int(round(y))

    x1 = clamp(x - radius, 0, w - 1)
    y1 = clamp(y - radius, 0, h - 1)
    x2 = clamp(x + radius, 0, w - 1)
    y2 = clamp(y + radius, 0, h - 1)

    crop = image[y1:y2 + 1, x1:x2 + 1].copy()

    local_x = x - x1
    local_y = y - y1

    cv2.circle(crop, (local_x, local_y), 10, COLOR_POINT, -1)
    cv2.circle(crop, (local_x, local_y), 16, COLOR_POINT_RING, 2)

    cv2.line(
        crop,
        (local_x - 25, local_y),
        (local_x + 25, local_y),
        COLOR_CROSS,
        2,
    )

    cv2.line(
        crop,
        (local_x, local_y - 25),
        (local_x, local_y + 25),
        COLOR_CROSS,
        2,
    )

    return crop


def pad_crops_to_same_size(crops):
    max_h = max(crop.shape[0] for crop in crops)
    max_w = max(crop.shape[1] for crop in crops)

    padded_crops = []

    for crop in crops:
        padded = np.full(
            (max_h, max_w, 3),
            COLOR_BACKGROUND,
            dtype=np.uint8,
        )

        h, w = crop.shape[:2]
        padded[:h, :w] = crop

        padded_crops.append(padded)

    return padded_crops


def add_corner_label(crop, label):
    out = crop.copy()

    cv2.rectangle(
        out,
        (0, 0),
        (260, 55),
        COLOR_LABEL_BG,
        -1,
    )

    cv2.putText(
        out,
        label,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )

    return out


def stitch_neighbourhoods_2x2(image, final_corners):
    top_left = add_corner_label(
        crop_corner_neighbourhood(
            image,
            final_corners["top_left"],
            NEIGHBOURHOOD_RADIUS,
        ),
        "top_left",
    )

    top_right = add_corner_label(
        crop_corner_neighbourhood(
            image,
            final_corners["top_right"],
            NEIGHBOURHOOD_RADIUS,
        ),
        "top_right",
    )

    bottom_left = add_corner_label(
        crop_corner_neighbourhood(
            image,
            final_corners["bottom_left"],
            NEIGHBOURHOOD_RADIUS,
        ),
        "bottom_left",
    )

    bottom_right = add_corner_label(
        crop_corner_neighbourhood(
            image,
            final_corners["bottom_right"],
            NEIGHBOURHOOD_RADIUS,
        ),
        "bottom_right",
    )

    crops = pad_crops_to_same_size([
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ])

    tl, tr, bl, br = crops

    crop_h, crop_w = tl.shape[:2]

    vertical_sep = make_separator(crop_h, SEPARATOR_SIZE)

    horizontal_sep = make_separator(
        SEPARATOR_SIZE,
        crop_w * 2 + SEPARATOR_SIZE,
    )

    top_row = np.hstack([
        tl,
        vertical_sep,
        tr,
    ])

    bottom_row = np.hstack([
        bl,
        vertical_sep,
        br,
    ])

    stitched = np.vstack([
        top_row,
        horizontal_sep,
        bottom_row,
    ])

    return stitched


def build_review_image(original, final_corners):
    full_with_corners = draw_final_corners_on_original(
        original,
        final_corners,
    )

    neighbourhoods_2x2 = stitch_neighbourhoods_2x2(
        original,
        final_corners,
    )

    target_height = max(
        full_with_corners.shape[0],
        neighbourhoods_2x2.shape[0],
    )

    full_with_corners = resize_to_height(
        full_with_corners,
        target_height,
    )

    neighbourhoods_2x2 = resize_to_height(
        neighbourhoods_2x2,
        target_height,
    )

    target_width = max(
        full_with_corners.shape[1],
        neighbourhoods_2x2.shape[1],
    )

    full_with_corners = pad_to_size(
        full_with_corners,
        target_width,
        target_height,
    )

    neighbourhoods_2x2 = pad_to_size(
        neighbourhoods_2x2,
        target_width,
        target_height,
    )

    middle_separator = make_separator(
        SEPARATOR_SIZE,
        target_width,
    )

    review = np.vstack([
        full_with_corners,
        middle_separator,
        neighbourhoods_2x2,
    ])

    return review


# ============================================================
# PROCESSING
# ============================================================

def process_single_image(image_path):
    original = load_image(image_path)

    final_corners, debug_info = detect_and_refine_corners(original)

    return original, final_corners, debug_info


def print_corner_refinement(debug_info):
    if debug_info is None:
        return

    if "before_refine" not in debug_info:
        return

    if "after_refine" not in debug_info:
        return

    refine_info = debug_info.get("refine_info", {})

    for corner in CORNER_ORDER:
        before = debug_info["before_refine"][corner]
        after = debug_info["after_refine"][corner]

        shift_x = after[0] - before[0]
        shift_y = after[1] - before[1]

        info = refine_info.get(corner, {})

        score = info.get("score", None)
        h_support = info.get("h_support", 0)
        v_support = info.get("v_support", 0)

        score_text = (
            "None"
            if score is None
            else f"{score:.2f}"
        )

        print(
            f"{corner:13s} "
            f"before={str(before):16s} "
            f"after={str(after):16s} "
            f"shift=({shift_x:+3d},{shift_y:+3d}) "
            f"h={h_support:3d} "
            f"v={v_support:3d} "
            f"score={score_text:>8s}"
        )


def process_review_image(image_path, shared_output_folder):
    image_name = image_path.name
    total_start = time.perf_counter()

    try:
        eval_start = time.perf_counter()

        original, final_corners, debug_info = process_single_image(
            image_path,
        )

        eval_time = seconds_since(eval_start)

        if final_corners is None:
            total_time = seconds_since(total_start)

            timing = {
                "eval_time": eval_time,
                "build_time": 0.0,
                "save_time": 0.0,
                "total_time": total_time,
            }

            return image_name, "failed", None, timing, debug_info

        build_start = time.perf_counter()

        review_image = build_review_image(
            original,
            final_corners,
        )

        build_time = seconds_since(build_start)

        output_name = f"{image_path.stem}{OUTPUT_SUFFIX}"
        output_path = shared_output_folder / output_name

        save_start = time.perf_counter()

        ok = cv2.imwrite(str(output_path), review_image)

        save_time = seconds_since(save_start)
        total_time = seconds_since(total_start)

        timing = {
            "eval_time": eval_time,
            "build_time": build_time,
            "save_time": save_time,
            "total_time": total_time,
        }

        if not ok:
            return image_name, "write_failed", output_path, timing, debug_info

        return image_name, "success", output_path, timing, debug_info

    except Exception as e:
        total_time = seconds_since(total_start)

        timing = {
            "eval_time": 0.0,
            "build_time": 0.0,
            "save_time": 0.0,
            "total_time": total_time,
        }

        return image_name, f"error: {e}", None, timing, None


# ============================================================
# MAIN
# ============================================================

def main():
    folder = get_input_folder()
    image_files = get_image_files(folder)

    results_root = get_results_root()
    shared_output_folder = prepare_shared_output_folder(results_root)

    worker_count = choose_worker_count()

    print("Input folder:", folder)
    print("Results folder:", results_root)
    print("Shared output folder:", shared_output_folder)
    print("Images found:", len(image_files))
    print("Threads:", worker_count)

    if len(image_files) == 0:
        return

    batch_start = time.perf_counter()

    all_results = {}
    all_timings = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}

        for image_path in image_files:
            future = executor.submit(
                process_review_image,
                image_path,
                shared_output_folder,
            )

            futures[future] = image_path

        for future in as_completed(futures):
            image_path = futures[future]
            image_name = image_path.name

            print()
            print("=" * 80)
            print("Image:", image_name)

            (
                result_image_name,
                status,
                output_path,
                timing,
                debug_info,
            ) = future.result()

            all_results[result_image_name] = status
            all_timings[result_image_name] = timing

            print_corner_refinement(debug_info)

            print(f"Corner detection time: {timing['eval_time']:.3f} sec")
            print(f"Build review time:     {timing['build_time']:.3f} sec")
            print(f"Save review time:      {timing['save_time']:.3f} sec")
            print(f"Total image time:      {timing['total_time']:.3f} sec")

            if status == "success":
                print("Saved:", output_path)
            else:
                print(status.upper())

    batch_time = seconds_since(batch_start)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for image_name, status in all_results.items():
        timing = all_timings[image_name]

        print(
            f"{image_name:30s} "
            f"{status:15s} "
            f"detect={timing['eval_time']:.3f}s "
            f"build={timing['build_time']:.3f}s "
            f"save={timing['save_time']:.3f}s "
            f"total={timing['total_time']:.3f}s"
        )

    print()
    print("=" * 80)
    print("TIMING TOTALS")
    print("=" * 80)

    print(
        f"Total corner detection time: "
        f"{sum(t['eval_time'] for t in all_timings.values()):.3f} sec"
    )

    print(
        f"Total review build time:     "
        f"{sum(t['build_time'] for t in all_timings.values()):.3f} sec"
    )

    print(
        f"Total save time:             "
        f"{sum(t['save_time'] for t in all_timings.values()):.3f} sec"
    )

    print(f"Actual batch wall time:      {batch_time:.3f} sec")


if __name__ == "__main__":
    main()