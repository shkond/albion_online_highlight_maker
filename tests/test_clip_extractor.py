"""
Unit tests for ClipExtractor
"""
import unittest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.clip_extractor import ClipExtractor


class TestClipExtractor(unittest.TestCase):
    """Test clip extraction and merging."""
    
    def setUp(self):
        """Set up test extractor."""
        self.extractor = ClipExtractor()
    
    def test_extractor_creation(self):
        """Test extractor can be created."""
        self.assertIsNotNone(self.extractor)
    
    def test_merge_clips_empty_list(self):
        """Test merging with empty clip list."""
        result = self.extractor.merge_clips([], 'output.mp4')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
