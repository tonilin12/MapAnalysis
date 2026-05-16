import sys
from pathlib import Path

import cv2

from map_detector_project.CornerFolder.corner_utils import detect_and_refine_corners


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

CORNER_ORDER = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
]


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def get_input_folder():
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = Path(__file__).resolve().parent / "test_maps"

    folder = folder.resolve()

    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

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


def print_final_corners(image_name, final_corners):
    print()
    print("=" * 80)
    print("Image:", image_name)

    if final_corners is None:
        print("Final corners: FAILED")
        return

    print("Final corners:")

    for corner_name in CORNER_ORDER:
        x, y = final_corners[corner_name]
        print(f"{corner_name:13s}: ({int(round(x))}, {int(round(y))})")


def process_image(image_path):
    image = load_image(image_path)

    final_corners, _ = detect_and_refine_corners(image)

    print_final_corners(
        image_path.name,
        final_corners,
    )


def main():
    folder = get_input_folder()
    image_files = get_image_files(folder)

    print("Input folder:", folder)
    print("Images found:", len(image_files))

    for image_path in image_files:
        try:
            process_image(image_path)
        except Exception as e:
            print()
            print("=" * 80)
            print("Image:", image_path.name)
            print("ERROR:", e)


if __name__ == "__main__":
    main()