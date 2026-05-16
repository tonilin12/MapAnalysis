import os
import re
import time
from pathlib import Path

# Must be BEFORE importing PaddleOCR / paddle
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR

from map_detector_project.OCRFolder.cut_top import process_image


# ============================================================
# CONFIG
# ============================================================

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

LEFT_CUT_RATIO = 0.35
RIGHT_CUT_RATIO = 0.35
TOP_CUT_RATIO = 0.60

OCR_SCALE = 2.5
OCR_PADDING = 30


SHEET_PATTERN = re.compile(
    r"""
    ([A-Z])
    -
    (0?[1-9]|[1-5][0-9]|60)
    -
    (0?[1-9]|[1-9][0-9]|1[0-3][0-9]|14[0-4])
    -
    ([A-Da-d])
    -
    ([A-Da-d])
    """,
    re.VERBOSE,
)


# ============================================================
# IO HELPERS
# ============================================================

def get_script_dir():
    return Path(__file__).resolve().parent


def make_output_folder(folder_name):
    output_folder = get_script_dir() / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def get_all_images(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    image_paths.sort(key=lambda p: p.name.lower())
    return image_paths


def safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    text = str(text)

    replacements = {
        " ": "",
        "_": "-",
        "—": "-",
        "–": "-",
        "−": "-",
        ".": "-",
        ",": "-",
        ":": "-",
        ";": "-",
        "|": "-",
        "/": "-",
        "\\": "-",
        "(": "-",
        ")": "-",
        "[": "-",
        "]": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"-+", "-", text)

    return text.strip("-").strip()


def normalize_identifier(identifier):
    parts = str(identifier).split("-")

    if len(parts) != 5:
        return str(identifier)

    first = parts[0].upper()
    second = str(int(parts[1]))
    third = str(int(parts[2]))
    fourth = parts[3].upper()
    fifth = parts[4].lower()

    return f"{first}-{second}-{third}-{fourth}-{fifth}"


def normalize_display_text(text):
    text = str(text)

    skip_values = {
        "NO_IMAGE",
        "NO_TEXT_FOUND",
        "NO_PATTERN_FOUND",
        "READ_FAILED",
    }

    if text in skip_values:
        return text

    if text.startswith("FAILED"):
        return text

    return normalize_identifier(text)


def extract_sheet_identifiers(text, return_cleaned=False):
    cleaned = clean_text(text)
    found = []

    for match in SHEET_PATTERN.finditer(cleaned):
        identifier = normalize_identifier(match.group(0))
        found.append(identifier)

    if return_cleaned:
        return found, cleaned

    return found


# ============================================================
# OCR IMAGE PREP
# ============================================================

def crop_final_ocr_region(img):
    h, w = img.shape[:2]

    x1 = int(w * LEFT_CUT_RATIO)
    x2 = int(w * (1.0 - RIGHT_CUT_RATIO))

    middle = img[:, x1:x2].copy()

    mh, mw = middle.shape[:2]

    y1 = int(mh * TOP_CUT_RATIO)

    return middle[y1:mh, :].copy()


def preprocess_for_ocr(img):
    h, w = img.shape[:2]

    resized = cv2.resize(
        img,
        (int(w * OCR_SCALE), int(h * OCR_SCALE)),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    processed = cv2.copyMakeBorder(
        processed,
        OCR_PADDING,
        OCR_PADDING,
        OCR_PADDING,
        OCR_PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return processed


# ============================================================
# PADDLE OCR
# ============================================================

def create_ocr(
    use_textline_orientation=False,
    verbose=True,
):
    if verbose:
        print("Loading PaddleOCR...")

    start = time.perf_counter()

    try:
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            lang="en",
        )

        if verbose:
            print(f"Loaded PP-OCRv5 in {time.perf_counter() - start:.3f} sec")

        return ocr

    except Exception as e:
        if verbose:
            print("PP-OCRv5 failed, using default PaddleOCR.")
            print(f"Reason: {e}")

    ocr = PaddleOCR(
        use_textline_orientation=use_textline_orientation,
        lang="en",
    )

    if verbose:
        print(f"Loaded default PaddleOCR in {time.perf_counter() - start:.3f} sec")

    return ocr


def run_paddle_ocr(
    ocr,
    img,
    return_raw_lines=False,
):
    if img is None:
        if return_raw_lines:
            return "NO_IMAGE", []
        return "NO_IMAGE"

    ocr_result = ocr.predict(img)

    detected_items = []
    raw_lines = []

    if not ocr_result:
        if return_raw_lines:
            return "NO_TEXT_FOUND", raw_lines
        return "NO_TEXT_FOUND"

    for page in ocr_result:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])

        for text, score in zip(texts, scores):
            score = float(score)

            identifiers, cleaned = extract_sheet_identifiers(
                text,
                return_cleaned=True,
            )

            raw_lines.append(
                {
                    "text": text,
                    "cleaned": cleaned,
                    "score": score,
                }
            )

            for identifier in identifiers:
                detected_items.append(
                    {
                        "identifier": identifier,
                        "score": score,
                    }
                )

    if not detected_items:
        if return_raw_lines:
            return "NO_PATTERN_FOUND", raw_lines
        return "NO_PATTERN_FOUND"

    detected_items.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    final_identifier = normalize_identifier(
        detected_items[0]["identifier"]
    )

    if return_raw_lines:
        return final_identifier, raw_lines

    return final_identifier


# ============================================================
# RESULT IMAGE HELPERS
# ============================================================

def resize_width_keep_aspect(img, target_width):
    h, w = img.shape[:2]

    if w == target_width:
        return img

    scale = target_width / w
    new_h = max(1, int(h * scale))

    return cv2.resize(
        img,
        (target_width, new_h),
        interpolation=cv2.INTER_CUBIC,
    )


def create_result_image(cropped_img, text):
    if len(cropped_img.shape) == 2:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    display_text = normalize_display_text(text)

    target_width = max(cropped_img.shape[1], 700)

    top_img = resize_width_keep_aspect(
        cropped_img,
        target_width,
    )

    text_row_height = 140

    text_img = np.full(
        (text_row_height, target_width, 3),
        255,
        dtype=np.uint8,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4

    label = str(display_text)

    (tw, th), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    while tw > target_width - 40 and font_scale > 0.5:
        font_scale -= 0.1

        (tw, th), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

    x = (target_width - tw) // 2
    y = (text_row_height + th) // 2

    cv2.putText(
        text_img,
        label,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return np.vstack([top_img, text_img])


# ============================================================
# HIGH-LEVEL PROCESSING
# ============================================================

def prepare_ocr_crop(image_path):
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))

    if img is None:
        return None

    result = process_image(img)

    cropped_img = result["cropped"]

    final_crop = crop_final_ocr_region(cropped_img)

    ocr_ready_img = preprocess_for_ocr(final_crop)

    return final_crop, ocr_ready_img


def detect_pattern_from_image(
    image_path,
    ocr,
    return_raw_lines=False,
):
    try:
        prepared = prepare_ocr_crop(image_path)

        if prepared is None:
            if return_raw_lines:
                return "READ_FAILED", []
            return "READ_FAILED"

        final_crop, ocr_ready_img = prepared

        return run_paddle_ocr(
            ocr,
            ocr_ready_img,
            return_raw_lines=return_raw_lines,
        )

    except Exception as e:
        if return_raw_lines:
            return f"FAILED: {e}", []
        return f"FAILED: {e}"


def process_image_with_output(
    image_path,
    output_folder,
    ocr,
):
    image_path = Path(image_path)
    output_folder = Path(output_folder)

    print("\n" + "=" * 70)
    print(f"Processing: {image_path.name}")
    print("=" * 70)

    start = time.perf_counter()

    try:
        prepared = prepare_ocr_crop(image_path)

        if prepared is None:
            return {
                "filename": image_path.name,
                "detected": "READ_FAILED",
                "output": "",
                "time": 0.0,
                "raw_lines": [],
            }

        final_crop, ocr_ready_img = prepared

        detected_text, raw_lines = run_paddle_ocr(
            ocr,
            ocr_ready_img,
            return_raw_lines=True,
        )

        detected_text = normalize_display_text(detected_text)

        merged_result = create_result_image(
            final_crop,
            detected_text,
        )

        output_name = f"{safe_stem(image_path)}_ocr_result.png"
        output_path = output_folder / output_name

        cv2.imwrite(str(output_path), merged_result)

        elapsed = time.perf_counter() - start

        print(f"Detected: {detected_text}")
        print(f"Saved   : {output_path}")
        print(f"Time    : {elapsed:.3f} sec")

        return {
            "filename": image_path.name,
            "detected": detected_text,
            "output": str(output_path),
            "time": elapsed,
            "raw_lines": raw_lines,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start

        print(f"FAILED: {image_path.name}")
        print(f"Reason: {e}")

        return {
            "filename": image_path.name,
            "detected": f"FAILED: {e}",
            "output": "",
            "time": elapsed,
            "raw_lines": [],
        }


def save_summary(output_folder, results, summary_filename="ocr_summary.txt"):
    summary_path = Path(output_folder) / summary_filename

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("filename\tdetected\toutput\ttime_sec\n")

        for item in results:
            f.write(
                f'{item["filename"]}\t'
                f'{item["detected"]}\t'
                f'{item["output"]}\t'
                f'{item["time"]:.3f}\n'
            )

    print("\n========== SUMMARY SAVED ==========")
    print(summary_path)