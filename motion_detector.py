# motion_detector.py
import cv2
import numpy as np
from datetime import datetime
import os
import logging

class MotionDetector:
    def __init__(self, min_area=500, threshold=25):
        """Initialize motion detector with given parameters."""
        self.min_area = min_area
        self.threshold = threshold
        self.previous_frame = None
        self.setup_logging()
        self.ensure_output_directory()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def ensure_output_directory(self):
        """Ensure the output directory exists."""
        self.output_dir = "detected_motions"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")

    def detect_motion(self, frame):
        """Detect motion in the given frame."""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame if needed
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, frame

        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        # Process contours
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update previous frame
        self.previous_frame = gray

        return motion_detected, frame

    def save_frame(self, frame):
        """Save the frame with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/motion_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Saved detected motion: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving frame: {str(e)}")
