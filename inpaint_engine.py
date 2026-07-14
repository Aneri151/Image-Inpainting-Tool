import os
import cv2
import heapq
import numpy as np


# Maximum image dimension for low-memory hosting such as a 512 MB instance.
MAX_IMAGE_DIMENSION = 1000


def resize_for_processing(image, max_dimension=MAX_IMAGE_DIMENSION):
    """
    Resize large images while preserving aspect ratio.
    This prevents excessive RAM usage during inpainting.
    """
    if image is None:
        raise ValueError("Image not found!")

    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image

    scale = max_dimension / float(max(height, width))

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def load_and_convert(image):
    if image is None:
        raise ValueError("Image not found!")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def create_white_mask(gray):
    _, mask = cv2.threshold(
        gray,
        250,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return (mask > 0).astype(np.uint8)


def directional_weight(px, py, qx, qy, gx_q, gy_q):
    dx = px - qx
    dy = py - qy

    r2 = float(dx * dx + dy * dy) + 1e-6
    r = np.sqrt(r2)

    dot = (
        dx * float(gx_q)
        + dy * float(gy_q)
    )

    return dot / (r2 * r)


def _save_image(path, image):
    """
    Safely save an image using OpenCV.
    """
    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    cv2.imwrite(path, image)


def _normalize_for_save(image):
    """
    Normalize floating-point data to 0-255 for visualization.
    """
    normalized = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(np.uint8)


def inpaint_custom(image, output_dir=None):

    # ---------------------------------------------------------
    # 1. Prepare image
    # ---------------------------------------------------------

    image = resize_for_processing(image)

    img_rgb = load_and_convert(image)

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_BGR2GRAY
    )

    mask = create_white_mask(gray)

    if output_dir is not None:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

        _save_image(
            os.path.join(
                output_dir,
                "original.png"
            ),
            img_rgb
        )

        _save_image(
            os.path.join(
                output_dir,
                "gray.png"
            ),
            gray
        )

        _save_image(
            os.path.join(
                output_dir,
                "mask.png"
            ),
            mask * 255
        )

    # ---------------------------------------------------------
    # 2. Use float32 instead of float64
    # ---------------------------------------------------------

    img = img_rgb.astype(
        np.float32
    )

    height, width, channels = img.shape

    KNOWN = 0
    BAND = 1
    INSIDE = 2

    state = np.full(
        (height, width),
        INSIDE,
        dtype=np.uint8
    )

    state[mask == 0] = KNOWN

    # ---------------------------------------------------------
    # 3. Distance transform
    # ---------------------------------------------------------

    dist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        3
    ).astype(np.float32)

    # ---------------------------------------------------------
    # 4. Create initial boundary queue
    # ---------------------------------------------------------

    pq = []

    for y in range(height):

        for x in range(width):

            if mask[y, x] != 1:
                continue

            is_boundary = (
                (y > 0 and mask[y - 1, x] == 0)
                or
                (
                    y < height - 1
                    and mask[y + 1, x] == 0
                )
                or
                (x > 0 and mask[y, x - 1] == 0)
                or
                (
                    x < width - 1
                    and mask[y, x + 1] == 0
                )
            )

            if is_boundary:

                state[y, x] = BAND

                heapq.heappush(
                    pq,
                    (
                        float(dist[y, x]),
                        y,
                        x
                    )
                )

    if output_dir is not None:

        distance_visual = _normalize_for_save(
            dist
        )

        distance_visual = cv2.applyColorMap(
            distance_visual,
            cv2.COLORMAP_JET
        )

        _save_image(
            os.path.join(
                output_dir,
                "distance_transform.png"
            ),
            distance_visual
        )

        del distance_visual

        state_map = np.zeros(
            (
                height,
                width,
                3
            ),
            dtype=np.uint8
        )

        state_map[state == BAND] = [
            255,
            0,
            0
        ]

        state_map[state == KNOWN] = [
            0,
            255,
            0
        ]

        state_map[state == INSIDE] = [
            0,
            0,
            255
        ]

        _save_image(
            os.path.join(
                output_dir,
                "state_map.png"
            ),
            state_map
        )

        del state_map

    # ---------------------------------------------------------
    # 5. Calculate gradients using float32
    # ---------------------------------------------------------

    gray_f = gray.astype(
        np.float32
    )

    gx = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    if output_dir is not None:

        gx_visual = _normalize_for_save(
            np.abs(gx)
        )

        gy_visual = _normalize_for_save(
            np.abs(gy)
        )

        grad_mag = cv2.magnitude(
            gx,
            gy
        )

        grad_visual = _normalize_for_save(
            grad_mag
        )

        combined = np.hstack(
            (
                gx_visual,
                gy_visual,
                grad_visual
            )
        )

        _save_image(
            os.path.join(
                output_dir,
                "gradients.png"
            ),
            combined
        )

        del gx_visual
        del gy_visual
        del grad_mag
        del grad_visual
        del combined

    # ---------------------------------------------------------
    # 6. Initial fast-marching style fill
    # ---------------------------------------------------------

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1)
    ]

    out = img.copy()

    if output_dir is not None:

        _save_image(
            os.path.join(
                output_dir,
                "mask_before_fill.png"
            ),
            mask * 255
        )

    while pq:

        _, y, x = heapq.heappop(pq)

        if state[y, x] == KNOWN:
            continue

        for channel in range(channels):

            weight_sum = 0.0
            intensity_sum = 0.0

            values = []

            for dy, dx in neighbors:

                ny = y + dy
                nx = x + dx

                if (
                    0 <= ny < height
                    and
                    0 <= nx < width
                    and
                    state[ny, nx] == KNOWN
                ):

                    weight = directional_weight(
                        x,
                        y,
                        nx,
                        ny,
                        gx[ny, nx],
                        gy[ny, nx]
                    )

                    if weight > 0:

                        weight_sum += weight

                        intensity_sum += (
                            weight
                            * float(
                                out[
                                    ny,
                                    nx,
                                    channel
                                ]
                            )
                        )

                    values.append(
                        float(
                            out[
                                ny,
                                nx,
                                channel
                            ]
                        )
                    )

            if weight_sum > 0:

                out[
                    y,
                    x,
                    channel
                ] = (
                    intensity_sum
                    /
                    weight_sum
                )

            elif values:

                out[
                    y,
                    x,
                    channel
                ] = np.median(values)

        state[y, x] = KNOWN

        for dy, dx in neighbors:

            ny = y + dy
            nx = x + dx

            if (
                0 <= ny < height
                and
                0 <= nx < width
                and
                state[ny, nx] == INSIDE
            ):

                state[
                    ny,
                    nx
                ] = BAND

                heapq.heappush(
                    pq,
                    (
                        float(
                            dist[
                                ny,
                                nx
                            ]
                        ),
                        ny,
                        nx
                    )
                )

    # Release arrays no longer needed.

    del pq
    del state
    del dist

    if output_dir is not None:

        _save_image(
            os.path.join(
                output_dir,
                "after_initial_fill.png"
            ),
            out
        )

    # ---------------------------------------------------------
    # 7. Memory-optimized diffusion
    # ---------------------------------------------------------

    mask_boolean = (
        mask == 1
    )

    gray_r = cv2.cvtColor(
        np.clip(
            out,
            0,
            255
        ).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    # Reduce from 40 iterations if needed for a 512 MB server.
    diffusion_iterations = 30

    for _ in range(
        diffusion_iterations
    ):

        grad_x = cv2.Sobel(
            gray_r,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        grad_y = cv2.Sobel(
            gray_r,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        grad = cv2.magnitude(
            grad_x,
            grad_y
        )

        del grad_x
        del grad_y

        conductivity = np.exp(
            -np.square(
                grad / 15.0
            )
        ).astype(np.float32)

        del grad

        for channel in range(
            channels
        ):

            current = out[
                ...,
                channel
            ]

            # One flux array instead of many
            # full-sized north/south/east/west arrays.

            flux = np.zeros_like(
                current,
                dtype=np.float32
            )

            # North
            flux[1:, :] += (
                (
                    conductivity[1:, :]
                    +
                    conductivity[:-1, :]
                )
                * 0.5
                *
                (
                    current[:-1, :]
                    -
                    current[1:, :]
                )
            )

            # South
            flux[:-1, :] += (
                (
                    conductivity[:-1, :]
                    +
                    conductivity[1:, :]
                )
                * 0.5
                *
                (
                    current[1:, :]
                    -
                    current[:-1, :]
                )
            )

            # West
            flux[:, 1:] += (
                (
                    conductivity[:, 1:]
                    +
                    conductivity[:, :-1]
                )
                * 0.5
                *
                (
                    current[:, :-1]
                    -
                    current[:, 1:]
                )
            )

            # East
            flux[:, :-1] += (
                (
                    conductivity[:, :-1]
                    +
                    conductivity[:, 1:]
                )
                * 0.5
                *
                (
                    current[:, 1:]
                    -
                    current[:, :-1]
                )
            )

            # Update only the masked region.

            current[
                mask_boolean
            ] += (
                0.18
                *
                flux[
                    mask_boolean
                ]
            )

            del flux

        del conductivity

        # Update grayscale only once per iteration,
        # not once for every color channel.

        temp_uint8 = np.clip(
            out,
            0,
            255
        ).astype(np.uint8)

        gray_r = cv2.cvtColor(
            temp_uint8,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        del temp_uint8

    # ---------------------------------------------------------
    # 8. Final result
    # ---------------------------------------------------------

    final_image = np.clip(
        out,
        0,
        255
    ).astype(np.uint8)

    if output_dir is not None:

        _save_image(
            os.path.join(
                output_dir,
                "final.png"
            ),
            final_image
        )

    # Release large processing arrays.

    del img
    del out
    del gray_f
    del gray_r
    del gx
    del gy

    return {
        "original": "original.png",
        "gray": "gray.png",
        "mask": "mask.png",
        "distance_transform": "distance_transform.png",
        "state_map": "state_map.png",
        "gradients": "gradients.png",
        "mask_before_fill": "mask_before_fill.png",
        "after_initial_fill": "after_initial_fill.png",
        "final": "final.png",
    }
