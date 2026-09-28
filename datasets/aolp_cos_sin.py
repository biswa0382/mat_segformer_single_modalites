#!/usr/bin/env python3
"""
Load an AoLP (Angle of Linear Polarization) .npy file, compute its
sin and cos components, and save results as both .npy and .png.

AoLP is periodic with period 180 degrees (pi radians), since a
polarization angle of theta is physically identical to theta + 180.
To respect this, we encode using sin(2*theta) and cos(2*theta)
rather than sin(theta)/cos(theta) directly.

Assumption about input encoding:
    The input is uint8. We assume it encodes AoLP in the range
    [0, 180) degrees, linearly scaled to [0, 255]:
        theta_deg = aolp_uint8 * (180.0 / 255.0)
    If your data uses a different scale (e.g. radians pi/255, or
    the raw pixel value already IS degrees/radians), adjust the
    `decode_aolp` function below accordingly.
"""

import numpy as np
import argparse
import os


def decode_aolp(aolp_raw: np.ndarray) -> np.ndarray:
    """Convert raw uint8 AoLP values to radians in [0, pi)."""
    aolp_raw = aolp_raw.astype(np.float64)
    theta_deg = aolp_raw * (180.0 / 255.0)
    theta_rad = np.deg2rad(theta_deg)
    return theta_rad


def aolp_cos_sin(aolp_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 cos(2*theta), sin(2*theta), in that channel order."""
    if not np.isfinite(aolp_raw).all() or np.any(aolp_raw < 0) or np.any(aolp_raw > 255):
        raise ValueError("AoLP must use the stored 0..255 angle encoding")
    theta_rad = decode_aolp(aolp_raw)
    return (
        np.cos(2.0 * theta_rad).astype(np.float32),
        np.sin(2.0 * theta_rad).astype(np.float32),
    )


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Compute AoLP sin/cos maps.")
    parser.add_argument(
        "--input", "-i",
        default="/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/mat_seg_dataset/set_3/0041/light_on/aligned/npy/polar_aolp.npy",
        help="Path to the input polar_aolp.npy file",
    )
    parser.add_argument(
        "--outdir", "-o",
        default=None,
        help="Output directory (defaults to same directory as input)",
    )
    args = parser.parse_args()

    in_path = args.input
    outdir = args.outdir or os.path.dirname(in_path)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_path))[0]

    aolp_raw = np.load(in_path)
    print(f"Loaded {in_path}: dtype={aolp_raw.dtype}, shape={aolp_raw.shape}, "
          f"min={aolp_raw.min()}, max={aolp_raw.max()}")

    aolp_cos, aolp_sin = aolp_cos_sin(aolp_raw)

    # --- Save .npy (float32, range [-1, 1]) ---
    sin_npy_path = os.path.join(outdir, f"{base}_sin.npy")
    cos_npy_path = os.path.join(outdir, f"{base}_cos.npy")
    np.save(sin_npy_path, aolp_sin)
    np.save(cos_npy_path, aolp_cos)
    print(f"Saved: {sin_npy_path}")
    print(f"Saved: {cos_npy_path}")

    # --- Save .png (normalized to [0, 255] for visualization) ---
    def to_uint8(arr):
        norm = (arr + 1.0) / 2.0  # [-1,1] -> [0,1]
        return (norm * 255.0).clip(0, 255).astype(np.uint8)

    sin_png_path = os.path.join(outdir, f"{base}_sin.png")
    cos_png_path = os.path.join(outdir, f"{base}_cos.png")
    plt.imsave(sin_png_path, to_uint8(aolp_sin), cmap="gray")
    plt.imsave(cos_png_path, to_uint8(aolp_cos), cmap="gray")
    print(f"Saved: {sin_png_path}")
    print(f"Saved: {cos_png_path}")


if __name__ == "__main__":
    main()
