%%writefile src/optical_flow.py
import cv2
import numpy as np


def compute_farneback_flow(prev_gray, next_gray):
    """
    Compute dense optical flow between two grayscale frames
    using OpenCV Farnebäck optical flow.
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow


def flow_to_hsv_rgb(flow):
    """
    Convert optical flow field to RGB visualization.

    Hue represents motion direction.
    Value represents motion magnitude.
    """

    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    # OpenCV HSV hue range: [0, 179]
    hsv[..., 0] = np.clip(ang * 180 / np.pi / 2, 0, 179).astype(np.uint8)

    # Saturation fixed to maximum
    hsv[..., 1] = 255

    # Value represents normalized motion magnitude
    hsv[..., 2] = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_rgb


def frames_to_flow_rgb(frames):
    """
    Convert a list of RGB video frames to optical-flow RGB maps.

    Parameters
    ----------
    frames : list of np.ndarray
        RGB frames, each with shape [H, W, 3].

    Returns
    -------
    flows : list of np.ndarray
        RGB optical flow maps. If input has T frames, output has T-1 flow maps.
    """

    if len(frames) < 2:
        return [np.zeros_like(frames[0])]

    gray_frames = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        for frame in frames
    ]

    flows = []

    for i in range(len(gray_frames) - 1):
        flow = compute_farneback_flow(
            gray_frames[i],
            gray_frames[i + 1]
        )

        flow_rgb = flow_to_hsv_rgb(flow)
        flows.append(flow_rgb)

    return flows