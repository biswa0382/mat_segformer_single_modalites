# from pathlib import Path

# import numpy as np
# from PIL import Image


# def inspect_image(image_path: str) -> None:
#     image_path = Path(image_path)

#     with Image.open(image_path) as img:
#         print("=" * 60)
#         print("IMAGE INFORMATION")
#         print("=" * 60)

#         print(f"Path       : {image_path}")
#         print(f"Format     : {img.format}")
#         print(f"Mode       : {img.mode}")
#         print(f"Size       : {img.size}")
#         print(f"Width      : {img.width}")
#         print(f"Height     : {img.height}")

#         array = np.asarray(img)

#     print(f"\nNumPy dtype : {array.dtype}")
#     print(f"Shape       : {array.shape}")
    
#     if array.ndim == 2:
#         print("\nThis is a single-channel grayscale image.")
#         print(f"Min value   : {array.min()}")
#         print(f"Max value   : {array.max()}")

#     elif array.ndim == 3:
#         channels = array.shape[2]

#         print(f"\nNumber of channels: {channels}")

#         for c in range(channels):
#             channel = array[:, :, c]

#             print(
#                 f"Channel {c}: "
#                 f"min={channel.min()}, "
#                 f"max={channel.max()}, "
#                 f"mean={channel.mean():.4f}, "
#                 f"std={channel.std():.4f}"
#             )

#         # Check whether all channels contain exactly the same values
#         all_same = all(
#             np.array_equal(array[:, :, 0], array[:, :, c])
#             for c in range(1, channels)
#         )

#         print("\n" + "-" * 60)
#         print(f"Are all channels exactly identical? : {all_same}")

#         if channels >= 2:
#             for c in range(1, channels):
#                 difference = np.abs(
#                     array[:, :, 0].astype(np.int64)
#                     - array[:, :, c].astype(np.int64)
#                 )

#                 print(
#                     f"Channel 0 vs Channel {c}: "
#                     f"max difference = {difference.max()}, "
#                     f"mean difference = {difference.mean():.6f}"
#                 )

#     else:
#         print("\nUnexpected image dimensions.")

#     print("=" * 60)


# if __name__ == "__main__":
#     image_path = "/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/mat_seg_dataset/set_1/0001/light_on/aligned/png/nir.png"

#     inspect_image(image_path)


import numpy as np


def inspect_npy_image(image_path):
    image = np.load(image_path, allow_pickle=False)

    print("=" * 60)
    print("NPY IMAGE INFORMATION")
    print("=" * 60)

    print(f"Path       : {image_path}")
    print(f"Dtype      : {image.dtype}")
    print(f"Shape      : {image.shape}")
    print(f"Dimensions : {image.ndim}")
    print(f"Min value  : {image.min()}")
    print(f"Max value  : {image.max()}")
    print(f"Mean       : {image.mean():.6f}")
    print(f"Std        : {image.std():.6f}")

    print("\n" + "-" * 60)

    if image.ndim == 2:
        print("Single-channel image")
        print(f"Height     : {image.shape[0]}")
        print(f"Width      : {image.shape[1]}")

    elif image.ndim == 3:
        print("Multi-channel image")

        print(f"Height     : {image.shape[0]}")
        print(f"Width      : {image.shape[1]}")
        print(f"Channels  : {image.shape[2]}")

        # Check every channel
        for c in range(image.shape[2]):
            channel = image[:, :, c]

            print(
                f"\nChannel {c}:"
                f"\n  Min   : {channel.min()}"
                f"\n  Max   : {channel.max()}"
                f"\n  Mean  : {channel.mean():.6f}"
                f"\n  Std   : {channel.std():.6f}"
            )

        # Check whether all channels are identical
        first_channel = image[:, :, 0]

        all_same = all(
            np.array_equal(first_channel, image[:, :, c])
            for c in range(1, image.shape[2])
        )

        print("\n" + "-" * 60)
        print(f"Are all channels identical? : {all_same}")

        # Pairwise comparison
        for c in range(1, image.shape[2]):
            diff = np.abs(
                first_channel.astype(np.float64)
                - image[:, :, c].astype(np.float64)
            )

            print(
                f"Channel 0 vs Channel {c}:"
                f"\n  Max difference  : {diff.max()}"
                f"\n  Mean difference : {diff.mean():.6f}"
            )

    else:
        print(f"Unexpected number of dimensions: {image.ndim}")

    print("=" * 60)


if __name__ == "__main__":

    image_path = "/media/voyager/7e280623-1817-45d4-9e64-b992b9c41c77/base_model/dataset/mat_seg_dataset/set_3/0041/light_on/aligned/npy/polar.npy"

    inspect_npy_image(image_path)