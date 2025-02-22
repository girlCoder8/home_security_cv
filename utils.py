# utils.py

import cv2
import logging

def setup_window(window_name="Motion Detection"):
    """Set up display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    return window_name

def cleanup():
    """Clean up OpenCV windows."""
    cv2.destroyAllWindows()

def resize_frame(frame, width=None, height=None):
    """Resize frame while maintaining aspect ratio."""
    if frame is None:
        return None

    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]
    if width is None:
        aspect_ratio = height / float(h)
        dim = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
