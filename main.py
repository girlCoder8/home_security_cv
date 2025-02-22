# main.py

import cv2
import logging
from camera import Camera
from motion_detector import MotionDetector
from utils import setup_window, cleanup, resize_frame

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize components
    camera = Camera()
    motion_detector = MotionDetector()
    window_name = setup_window()

    # Start camera
    if not camera.start():
        logger.error("Failed to start camera. Exiting.")
        return

    logger.info("Intruder detection system started. Press 'q' to quit.")

    try:
        while True:
            # Read frame
            frame = camera.read_frame()
            if frame is None:
                logger.error("Failed to read frame")
                break

            # Resize frame for display
            display_frame = resize_frame(frame, width=800)

            # Detect motion
            motion_detected, processed_frame = motion_detector.detect_motion(display_frame)

            # Save frame if motion detected
            if motion_detected:
                motion_detector.save_frame(frame)
                logger.info("Motion detected!")

            # Display frame
            cv2.imshow(window_name, processed_frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit command received")
                break

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

    finally:
        # Cleanup
        camera.release()
        cleanup()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
