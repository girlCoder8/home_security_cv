# camera.py

import cv2
import logging
from datetime import datetime

class Camera:
    def __init__(self, camera_id=0):
        """Initialize camera capture."""
        self.camera_id = camera_id
        self.cap = None
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            self.logger.info("Camera started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error starting camera: {str(e)}")
            return False

    def read_frame(self):
        """Read a frame from the camera."""
        if self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read frame")
                return None
            return frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return None

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera released")
