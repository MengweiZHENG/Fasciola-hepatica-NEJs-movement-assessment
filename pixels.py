import cv2
import numpy as np
import argparse
import os
import re
from pathlib import Path

def natural_key(s):
    """Sort like humans: file2 < file10."""
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def gather_tiff_paths(tiff_folder: str):
    """Return sorted list of .tif/.tiff files, case-insensitive."""
    p = Path(tiff_folder)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {tiff_folder}")
    exts = {'.tif', '.tiff'}
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files = sorted(files, key=lambda x: natural_key(x.name))
    return [str(f) for f in files]

def read_gray_8bit(path: str):
    """
    Read image robustly:
    - keep original depth
    - convert to grayscale if needed
    - normalize to 8-bit for stable video writing/processing
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        minv, maxv = int(np.min(img)), int(np.max(img))
        if maxv > minv:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    return img

def select_codec(video_path: str):
    """Pick codec based on file extension."""
    ext = Path(video_path).suffix.lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    elif ext == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    else:
        raise ValueError(f"Unsupported video extension '{ext}'. Use .avi or .mp4")

def process_tiff_folder(
    tiff_folder,
    output_file,
    output_video,
    background_image,
    heatmap_image,
    threshold=30,
    fps=10
):
    tiff_files = gather_tiff_paths(tiff_folder)
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_folder}")

    first_frame = read_gray_8bit(tiff_files[0])
    height, width = first_frame.shape
    avg_frame = np.float32(first_frame)
    movement_heatmap = np.zeros_like(first_frame, dtype=np.uint32)
    change_counts = []

    # Video writer
    codec = select_codec(output_video)
    video_writer = cv2.VideoWriter(output_video, codec, fps, (width, height), isColor=True)
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{output_video}'. Try a different codec or extension.")

    alpha = 0.01

    for frame_path in tiff_files:
        try:
            gray = read_gray_8bit(frame_path)
        except Exception as e:
            print(f"Warning: {e}. Skipping.")
            continue

        video_writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        change_mask = diff > threshold
        changed_pixels = int(np.sum(change_mask))
        change_counts.append(changed_pixels)

        movement_heatmap += change_mask.astype(np.uint8)
        avg_frame = cv2.addWeighted(gray.astype(np.float32), alpha, avg_frame, 1 - alpha, 0)

    video_writer.release()

    # Save pixel-change stats
    with open(output_file, 'w') as f:
        f.write("#seconds\tchanged_pixels\trate_per_second\n")
        for i, count in enumerate(change_counts):
            seconds = i // fps
            f.write(f"{seconds}\t{count}\t{count/fps:.2f}\n")

    # Save background
    cv2.imwrite(background_image, cv2.convertScaleAbs(avg_frame))

    # Save heatmap (convert uint32 → float32 for normalization)
    heatmap_float = movement_heatmap.astype(np.float32)
    heatmap_norm = cv2.normalize(heatmap_float, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_image, heatmap_colored)

    print(f"Done. Output written to {output_file}")
    print(f"Video saved to {output_video}")
    print(f"Background image saved to {background_image}")
    print(f"Heatmap image saved to {heatmap_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIFF stack pixel change detector")
    parser.add_argument("-i", "--input-folder", required=True, help="Folder containing TIFF files")
    parser.add_argument("-o", "--output", required=True, help="Output text report file")
    parser.add_argument("--video-output", required=True, help="Output video filename (.avi or .mp4)")
    parser.add_argument("--background-image", required=True, help="Filename for background image")
    parser.add_argument("--heatmap-image", required=True, help="Filename for movement heatmap image")
    parser.add_argument("--threshold", type=int, default=30, help="Pixel change threshold (default: 30)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    args = parser.parse_args()

    process_tiff_folder(
        tiff_folder=args.input_folder,
        output_file=args.output,
        output_video=args.video_output,
        background_image=args.background_image,
        heatmap_image=args.heatmap_image,
        threshold=args.threshold,
        fps=args.fps
    )