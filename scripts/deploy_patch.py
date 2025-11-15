"""
Script for physical deployment of adversarial patches.
"""
import torch
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path


def print_patch(patch_path: str, output_path: str, size_cm: tuple = (10, 10), dpi: int = 300):
    """
    Prepare patch for printing.
    
    Args:
        patch_path: Path to saved patch
        output_path: Output path for printable image
        size_cm: Size in centimeters (width, height)
        dpi: DPI for printing
    """
    # Load patch
    patch = torch.load(patch_path)
    if isinstance(patch, np.ndarray):
        patch_np = patch
    else:
        patch_np = patch.cpu().numpy()
    
    # Convert to (H, W, C) format
    if len(patch_np.shape) == 3 and patch_np.shape[0] == 3:
        patch_np = patch_np.transpose(1, 2, 0)
    
    # Normalize to [0, 255]
    if patch_np.max() <= 1.0:
        patch_np = (patch_np * 255).astype(np.uint8)
    else:
        patch_np = patch_np.astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(patch_np)
    
    # Resize to print size
    width_px = int(size_cm[0] * dpi / 2.54)
    height_px = int(size_cm[1] * dpi / 2.54)
    img = img.resize((width_px, height_px), Image.LANCZOS)
    
    # Save
    img.save(output_path, dpi=(dpi, dpi))
    print(f"Patch saved for printing: {output_path}")
    print(f"Size: {size_cm[0]}cm x {size_cm[1]}cm at {dpi} DPI")


def capture_video_with_patch(
    patch_path: str,
    output_path: str,
    duration: int = 10,
    camera_id: int = 0
):
    """
    Capture video with physical patch (simulated).
    
    Args:
        patch_path: Path to patch
        output_path: Output video path
        duration: Duration in seconds
        camera_id: Camera device ID
    """
    print("Note: This is a simulation. In practice, you would:")
    print("1. Print the patch")
    print("2. Place it in the scene")
    print("3. Capture video with camera")
    print("4. Process video through pipeline")
    
    # Load patch
    patch = torch.load(patch_path)
    if isinstance(patch, np.ndarray):
        patch_np = patch
    else:
        patch_np = patch.cpu().numpy()
    
    # Simulate video capture
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}. Using simulated video.")
        # Create dummy video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        
        for _ in range(duration * 30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        print(f"Simulated video saved: {output_path}")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        
        frame_count = 0
        while frame_count < duration * 30:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frame_count += 1
            else:
                break
        
        cap.release()
        out.release()
        print(f"Video captured: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy adversarial patch')
    parser.add_argument('--patch', required=True, help='Path to patch file')
    parser.add_argument('--mode', choices=['print', 'capture'], default='print',
                       help='Deployment mode')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--size', nargs=2, type=float, default=(10, 10),
                       help='Size in cm (width height)')
    parser.add_argument('--dpi', type=int, default=300, help='Print DPI')
    parser.add_argument('--duration', type=int, default=10, help='Video duration (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 'print':
        print_patch(args.patch, args.output, tuple(args.size), args.dpi)
    elif args.mode == 'capture':
        capture_video_with_patch(args.patch, args.output, args.duration)


