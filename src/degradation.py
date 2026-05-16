%%writefile src/degradation.py
import cv2


def degrade_frame(frame, level="strong"):
    """
    Deterministic visual degradation.

    目的：
    削弱形状、纹理、边缘等 RGB 外观线索，
    但不引入随机噪声，从而保证每次评估结果一致。

    Parameters
    ----------
    frame : np.ndarray
        RGB image frame, shape = [H, W, 3].
    level : str
        One of: "none", "mild", "strong", "extreme".

    Returns
    -------
    degraded : np.ndarray
        Degraded RGB image frame.
    """

    if level == "none":
        return frame

    h, w = frame.shape[:2]

    if level == "mild":
        factor = 6
        kernel = 9
        alpha = 0.85
    elif level == "strong":
        factor = 16
        kernel = 21
        alpha = 0.65
    elif level == "extreme":
        factor = 24
        kernel = 31
        alpha = 0.55
    else:
        raise ValueError(
            "level must be one of: none, mild, strong, extreme"
        )

    # 1. 强制低分辨率化
    small_w = max(8, w // factor)
    small_h = max(8, h // factor)

    small = cv2.resize(
        frame,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA
    )

    degraded = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # 2. 高斯模糊，进一步削弱边缘和形状
    if kernel % 2 == 0:
        kernel += 1

    degraded = cv2.GaussianBlur(
        degraded,
        (kernel, kernel),
        sigmaX=0
    )

    # 3. 对比度压缩
    degraded = cv2.convertScaleAbs(
        degraded,
        alpha=alpha,
        beta=10
    )

    return degraded