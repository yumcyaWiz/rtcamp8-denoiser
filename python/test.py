import numpy as np
import cv2 as cv


def clamp(x, xmin, xmax):
    return min(max(x, xmin), xmax)


def get_image_idx(i: int, j: int, width: int, height: int) -> tuple[int, int]:
    return (clamp(i, 0, width - 1), clamp(j, 0, height - 1))


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    return np.exp(-np.sqrt(np.dot(x - y, x - y)) / (2 * sigma**2))


# Bitterli, Benedikt, et al. "Nonlinearly weighted first‐order regression for denoising Monte Carlo renderings." Computer Graphics Forum. Vol. 35. No. 4. 2016.
if __name__ == "__main__":
    K = 1
    sigma_b = 1.0

    beauty = cv.imread("../assets/cornell_1spp/color.hdr",
                       flags=cv.IMREAD_UNCHANGED)
    albedo = cv.imread("../assets/cornell_1spp/albedo.hdr",
                       flags=cv.IMREAD_UNCHANGED)
    normal = cv.imread("../assets/cornell_1spp/normal.hdr",
                       flags=cv.IMREAD_UNCHANGED)

    width = beauty.shape[0]
    height = beauty.shape[1]

    denoised = np.empty((width, height, 3))
    for j in range(height):
        for i in range(width):
            idx0 = get_image_idx(i, j, width, height)

            x_width = 2 * K + 1
            x_height = 2 * K + 1
            x = np.empty((x_width * x_height, 6))
            x_idx = 0

            wx = np.empty((x_width * x_height))

            y = np.empty((x_width * x_height, 3))

            for v in range(-K, K + 1):
                for u in range(-K, K + 1):
                    idx1 = get_image_idx(i + u, j + v, width, height)
                    x[x_idx, :3] = albedo[idx1[0], idx1[1]] - \
                        albedo[idx0[0], idx0[1]]
                    x[x_idx, 3:6] = normal[idx1[0], idx1[1]] - \
                        normal[idx0[0], idx0[1]]

                    wx[x_idx] = gaussian_kernel(
                        beauty[idx0[0], idx0[1]
                               ], beauty[idx1[0], idx1[1]], sigma_b)

                    y[x_idx] = beauty[idx1[0], idx1[1]]

                    x_idx += 1

            X = np.concatenate(
                (np.ones(x_width * x_height).reshape((x_width * x_height, 1)), x), axis=1)
            W = np.diag(wx)
            eps = np.diag(0.00001 * np.ones(7))

            beta = np.linalg.inv(X.T @ W @ X + eps) @ X.T @ W @ y
            denoised[i, j] = beta[0]

    cv.imwrite("denoised.hdr", denoised)
