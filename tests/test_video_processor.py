"""
Unit tests for VideoProcessor
"""
import unittest
import numpy as np
from src.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    """Test video processing utilities."""
    
    def test_extract_skill_slots(self):
        """Test skill slot extraction."""
        # Create dummy 1080p frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Extract slots
        slots = VideoProcessor.extract_skill_slots(frame)
        
        # Check we get 8 slots
        self.assertEqual(len(slots), 8)
        
        # Check each slot has correct dimensions
        for slot in slots:
            self.assertEqual(slot.shape, (64, 64, 3))
    
    def test_is_red_screen_not_red(self):
        """Test red screen detection with non-red frame."""
        # Create blue frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # Blue channel (BGR format)
        
        is_red = VideoProcessor.is_red_screen(frame)
        self.assertFalse(is_red)
    
    def test_is_red_screen_red(self):
        """Test red screen detection with red frame."""
        # Create red frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # Red channel (BGR format)
        
        is_red = VideoProcessor.is_red_screen(frame)
        self.assertTrue(is_red)
    
    def test_is_red_screen_mixed(self):
        """Test red screen detection with mixed colors."""
        # Create frame with equal RGB
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
        
        is_red = VideoProcessor.is_red_screen(frame)
        self.assertFalse(is_red)


if __name__ == '__main__':
    unittest.main()
