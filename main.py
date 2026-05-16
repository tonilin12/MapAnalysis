import sys
import time
from pathlib import Path

import cv2

from map_detector_project.OCRFolder.ocr_utils import (
    get_all_images,
    create_ocr,
    detect_pattern_from_image,
)

from map_detector_project.CornerFolder.corner_utils import (
    detect_and_refine_corners,
)


# ============================================================
# CONFIG
# ============================================================

CORNER_ORDER = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]

SCRIPT_DIR = Path(__file__).resolve().parent
TXT_OUTPUT_PATH = SCRIPT_DIR / "combined_ocr_corner_results.txt"


# ============================================================
# TIMING HELPER
# ============================================================

def seconds_since(start_time):
    return time.perf_counter() - start_time


# ============================================================
# IMAGE HELPER
# ============================================================

def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


# ============================================================
# CORNER DETECTION
# ============================================================

def find_corners_for_image(image):
    final_corners, _ = detect_and_refine_corners(image)
    return final_corners


# ============================================================
# TXT HELPER
# ============================================================

def save_txt_results(results):
    with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            corners = r["final_corners"]

            if corners is None:
                continue

            line = [
                r["image_name"],

                round(corners["top_left"][0]),
                round(corners["top_left"][1]),

                round(corners["top_right"][0]),
                round(corners["top_right"][1]),

                round(corners["bottom_right"][0]),
                round(corners["bottom_right"][1]),

                round(corners["bottom_left"][0]),
                round(corners["bottom_left"][1]),

                r["detected_text"],
            ]

            f.write(",".join(map(str, line)) + "\n")

    print()
    print("=" * 80)
    print("TXT saved to:")
    print(TXT_OUTPUT_PATH)


# ============================================================
# PRINT HELPERS
# ============================================================

def print_final_corners(final_corners):
    if final_corners is None:
        print("Final corners: FAILED")
        return

    print("Final corners:")

    for corner_name in CORNER_ORDER:
        x, y = final_corners[corner_name]
        print(f"  {corner_name:13s}: x={x:.2f}, y={y:.2f}")


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(image_path, ocr):
    image_path = Path(image_path)

    combined_start = time.perf_counter()

    result = {
        "image_name": image_path.name,
        "detected_text": None,
        "final_corners": None,
        "combined_time_sec": None,
        "success": False,
        "error": None,
    }

    print()
    print("=" * 80)
    print("Processing:", image_path.name)

    try:
        # ----------------------------------------------------
        # OCR PART - unchanged
        # ----------------------------------------------------
        detected_text = detect_pattern_from_image(
            image_path=image_path,
            ocr=ocr,
            return_raw_lines=False,
        )

        result["detected_text"] = detected_text

        print("OCR detected:", detected_text)

        # ----------------------------------------------------
        # CORNER FIND PART
        # ----------------------------------------------------
        image = load_image(image_path)

        final_corners = find_corners_for_image(image)

        result["final_corners"] = final_corners
        result["success"] = final_corners is not None

        print_final_corners(final_corners)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

        print("ERROR:", e)

    combined_time = seconds_since(combined_start)

    result["combined_time_sec"] = round(combined_time, 6)

    print(f"Combined image time: {combined_time:.3f} sec")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python unified_script.py <folder_path>")
        return

    folder = Path(sys.argv[1]).resolve()

    if not folder.is_dir():
        print(f"Folder does not exist: {folder}")
        return

    image_paths = get_all_images(folder)

    print("=" * 80)
    print("UNIFIED OCR + CORNER DETECTION")
    print("=" * 80)

    print("Input folder :", folder)
    print("Images found :", len(image_paths))
    print("TXT output   :", TXT_OUTPUT_PATH)

    if not image_paths:
        print(f"No images found in folder: {folder}")
        return

    print()
    print("Creating OCR model...")

    ocr_create_start = time.perf_counter()

    ocr = create_ocr(
        use_textline_orientation=False,
        verbose=True,
    )

    ocr_create_time = seconds_since(ocr_create_start)

    print(f"OCR model creation time: {ocr_create_time:.3f} sec")

    batch_start = time.perf_counter()

    results = []

    for image_path in image_paths:
        image_result = process_one_image(
            image_path=image_path,
            ocr=ocr,
        )

        results.append(image_result)

    batch_time = seconds_since(batch_start)

    success_count = sum(
        1 for r in results if r["success"]
    )

    failed_count = len(results) - success_count

    save_txt_results(results)

    print()
    print("=" * 80)
    print("DONE")
    print("Success:", success_count)
    print("Failed :", failed_count)
    print(f"Batch processing time: {batch_time:.3f} sec")
    print(f"Total time with OCR model: {ocr_create_time + batch_time:.3f} sec")


if __name__ == "__main__":
    main()