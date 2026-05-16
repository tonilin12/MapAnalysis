import sys
from pathlib import Path

from map_detector_project.OCRFolder.ocr_utils import (
    get_all_images,
    create_ocr,
    detect_pattern_from_image,
)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(image_path, ocr):
    image_path = Path(image_path)

    print(f"Processing: {image_path.name}", flush=True)

    detected_text = detect_pattern_from_image(
        image_path=image_path,
        ocr=ocr,
        return_raw_lines=False,
    )

    return detected_text


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python script20.py <folder_path>")
        return

    folder = sys.argv[1]

    image_paths = get_all_images(folder)

    if not image_paths:
        print(f"No images found in folder: {folder}")
        return

    ocr = create_ocr(
        use_textline_orientation=False,
        verbose=True,
    )

    for image_path in image_paths:
        detected = process_one_image(
            image_path=image_path,
            ocr=ocr,
        )

        print(f"{image_path.name} -> {detected}", flush=True)


if __name__ == "__main__":
    main()