import os
import cv2
import heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from google.colab.patches import cv2_imshow  # noqa: F401
except ImportError:
    cv2_imshow = None


def load_and_convert(image):
    if image is None:
        raise ValueError("Image not found!")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def create_white_mask(gray):
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask > 0).astype(np.uint8)


def directional_weight(px, py, qx, qy, gx_q, gy_q):
    dx = px - qx
    dy = py - qy
    r2 = dx * dx + dy * dy + 1e-6
    r = np.sqrt(r2)
    dot = dx * gx_q + dy * gy_q
    return dot / (r2 * r)


def _save_rgb_image(path, bgr_image):
    cv2.imwrite(path, bgr_image)


def _save_gray_image(path, image):
    normalized = image
    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    cv2.imwrite(path, normalized)


def _save_colormap(path, image, cmap='jet'):
    plt.figure(figsize=(6, 6), dpi=120)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def _save_combined_gradients(path, gx, gy, grad_mag):
    plt.figure(figsize=(15, 4), dpi=120)
    plt.subplot(1, 3, 1)
    plt.imshow(gx, cmap='gray')
    plt.title('gx')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gy, cmap='gray')
    plt.title('gy')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(grad_mag, cmap='inferno')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def inpaint_custom(image, output_dir=None):
    img_rgb = load_and_convert(image)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    mask = create_white_mask(gray)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_rgb_image(os.path.join(output_dir, 'original.png'), img_rgb)
        _save_gray_image(os.path.join(output_dir, 'gray.png'), gray)
        _save_gray_image(os.path.join(output_dir, 'mask.png'), mask * 255)

    img = img_rgb.astype(np.float64)
    H, W, C = img.shape

    KNOWN, BAND, INSIDE = 0, 1, 2
    state = np.full((H, W), INSIDE, np.uint8)
    state[mask == 0] = KNOWN

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)

    pq = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] == 1:
                if ((y > 0 and mask[y - 1, x] == 0) or (y < H - 1 and mask[y + 1, x] == 0) or
                        (x > 0 and mask[y, x - 1] == 0) or (x < W - 1 and mask[y, x + 1] == 0)):
                    state[y, x] = BAND
                    heapq.heappush(pq, (dist[y, x], y, x))

    if output_dir is not None:
        _save_colormap(os.path.join(output_dir, 'distance_transform.png'), dist, cmap='jet')
        band_vis = np.zeros((H, W, 3), dtype=np.uint8)
        band_vis[state == BAND] = [255, 0, 0]
        band_vis[state == KNOWN] = [0, 255, 0]
        band_vis[state == INSIDE] = [0, 0, 255]
        _save_rgb_image(os.path.join(output_dir, 'state_map.png'), band_vis)

    gray_f = gray.astype(np.float64)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    if output_dir is not None:
        _save_combined_gradients(os.path.join(output_dir, 'gradients.png'), gx, gy, grad_mag)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = img.copy()

    if output_dir is not None:
        _save_gray_image(os.path.join(output_dir, 'mask_before_fill.png'), mask * 255)

    while pq:
        _, y, x = heapq.heappop(pq)
        if state[y, x] == KNOWN:
            continue

        for c in range(C):
            Wsum = 0.0
            Isum = 0.0
            vals = []

            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W and state[ny, nx] == KNOWN:
                    w = directional_weight(x, y, nx, ny, gx[ny, nx], gy[ny, nx])
                    if w > 0:
                        Wsum += w
                        Isum += w * out[ny, nx, c]
                    vals.append(out[ny, nx, c])

            out[y, x, c] = Isum / Wsum if Wsum > 0 else np.median(vals)

        state[y, x] = KNOWN

        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < H and 0 <= nx < W and state[ny, nx] == INSIDE:
                state[ny, nx] = BAND
                heapq.heappush(pq, (dist[ny, nx], ny, nx))

    if output_dir is not None:
        _save_rgb_image(os.path.join(output_dir, 'after_initial_fill.png'), np.clip(out, 0, 255).astype(np.uint8))

    mask_f = mask.copy()
    gray_r = gray_f.copy()

    for _ in range(40):
        gx = cv2.Sobel(gray_r, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_r, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        c = np.exp(-(grad / 15) ** 2)

        for k in range(3):
            I = out[..., k]

            north = np.zeros_like(I)
            north[1:, :] = I[:-1, :]
            south = np.zeros_like(I)
            south[:-1, :] = I[1:, :]
            east = np.zeros_like(I)
            east[:, :-1] = I[:, 1:]
            west = np.zeros_like(I)
            west[:, 1:] = I[:, :-1]

            cN = np.zeros_like(c)
            cN[1:, :] = (c[1:, :] + c[:-1, :]) * 0.5
            cS = np.zeros_like(c)
            cS[:-1, :] = (c[1:, :] + c[:-1, :]) * 0.5
            cW = np.zeros_like(c)
            cW[:, 1:] = (c[:, 1:] + c[:, :-1]) * 0.5
            cE = np.zeros_like(c)
            cE[:, :-1] = (c[:, 1:] + c[:, :-1]) * 0.5

            flux = cN * (north - I) + cS * (south - I) + cW * (west - I) + cE * (east - I)
            I_new = I + 0.18 * flux
            out[..., k][mask_f == 1] = I_new[mask_f == 1]

            gray_r = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    final_image = np.clip(out, 0, 255).astype(np.uint8)
    if output_dir is not None:
        _save_rgb_image(os.path.join(output_dir, 'final.png'), final_image)

    return {
        'original': 'original.png',
        'gray': 'gray.png',
        'mask': 'mask.png',
        'distance_transform': 'distance_transform.png' if False else 'distance_transform.png',
        'state_map': 'state_map.png',
        'gradients': 'gradients.png',
        'mask_before_fill': 'mask_before_fill.png',
        'after_initial_fill': 'after_initial_fill.png',
        'final': 'final.png',
    }
