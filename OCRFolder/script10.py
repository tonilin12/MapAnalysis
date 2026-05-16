
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

from map_detector_project.OCRFolder.ocr_utils import (
    create_ocr,
    get_all_images,
    process_image_with_output,
    save_summary,
)


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FOLDER_NAME = "final_ocr_result_all_images"

# Use 1 if you want PaddleOCR to load only once.
NUM_WORKERS = 1

WORKER_OCR = None


# ============================================================
# WORKER SETUP
# ============================================================

def init_worker():
    global WORKER_OCR

    if WORKER_OCR is None:
        WORKER_OCR = create_ocr(
            use_textline_orientation=True,
            verbose=True,
        )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_one_image_worker(args):
    global WORKER_OCR

    image_path, output_folder = args

    if WORKER_OCR is None:
        init_worker()

    return process_image_with_output(
        image_path=image_path,
        output_folder=output_folder,
        ocr=WORKER_OCR,
    )


# ============================================================
# OUTPUT FOLDER
# ============================================================

def create_clean_output_folder():
    output_folder = (
        Path(__file__).resolve().parent
        / OUTPUT_FOLDER_NAME
    )

    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_folder


# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.perf_counter()

    if len(sys.argv) < 2:
        print("Usage:")
        print("python script10.py <folder_path>")
        return

    folder = sys.argv[1]

    output_folder = create_clean_output_folder()

    image_paths = get_all_images(folder)

    if not image_paths:
        print(f"No images found in folder: {folder}")
        return

    print(f"Found {len(image_paths)} images.")

    workers = min(
        NUM_WORKERS,
        len(image_paths),
    )

    print(f"Using {workers} worker process(es).")

    tasks = [
        (image_path, output_folder)
        for image_path in image_paths
    ]

    # ------------------------------------------------------------
    # IMPORTANT:
    # If workers == 1, PaddleOCR loads only once.
    # If workers > 1, each process loads its own PaddleOCR model.
    # ------------------------------------------------------------

    if workers <= 1:
        init_worker()

        results = [
            process_one_image_worker(task)
            for task in tasks
        ]

    else:
        with mp.Pool(
            processes=workers,
            initializer=init_worker,
        ) as pool:
            results = pool.map(
                process_one_image_worker,
                tasks,
            )

    save_summary(
        output_folder=output_folder,
        results=results,
    )

    print("\n========== FINAL RESULTS ==========")

    for item in results:
        print(
            f'{item["filename"]} -> '
            f'{item["detected"]}'
        )

    total_time = time.perf_counter() - total_start

    print("\n========== TOTAL TIME ==========")
    print(f"Total runtime: {total_time:.3f} sec")

    print("\n========== OUTPUT FOLDER ==========")
    print(output_folder)


if __name__ == "__main__":
    mp.freeze_support()
    main()