"""
Unit tests for CombatDetector
"""
import unittest
from unittest.mock import Mock, MagicMock
from src.combat_detector import CombatDetector
from src.skill_slot_classifier import SkillSlotClassifier


class TestCombatDetector(unittest.TestCase):
    """Test combat detection logic."""
    
    def setUp(self):
        """Set up test detector."""
        # Create mock classifier
        self.classifier = Mock(spec=SkillSlotClassifier)
        self.detector = CombatDetector(self.classifier)
    
    def test_is_mounted_true(self):
        """Test mounted detection with 4+ empty slots."""
        slot_classes = ['Normal', 'Normal', 'Empty', 'Empty', 'Empty', 'Empty', 'Normal', 'Normal']
        self.assertTrue(self.detector.is_mounted(slot_classes))
    
    def test_is_mounted_false(self):
        """Test mounted detection with less than 4 empty slots."""
        slot_classes = ['Normal', 'Cooldown', 'Empty', 'Empty', 'Empty', 'Normal', 'Normal', 'Normal']
        self.assertFalse(self.detector.is_mounted(slot_classes))
    
    def test_has_cooldown_true(self):
        """Test cooldown detection with cooldown slots."""
        slot_classes = ['Normal', 'Cooldown', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal']
        self.assertTrue(self.detector.has_cooldown(slot_classes))
    
    def test_has_cooldown_false(self):
        """Test cooldown detection without cooldown slots."""
        slot_classes = ['Normal', 'Normal', 'Empty', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal']
        self.assertFalse(self.detector.has_cooldown(slot_classes))
    
    def test_cooldown_start_frames_constant(self):
        """Test that COOLDOWN_START_FRAMES is 3."""
        self.assertEqual(self.detector.COOLDOWN_START_FRAMES, 3)
    
    def test_no_cooldown_end_seconds_constant(self):
        """Test that NO_COOLDOWN_END_SECONDS is 60."""
        self.assertEqual(self.detector.NO_COOLDOWN_END_SECONDS, 60)
    
    def test_clip_buffer_constants(self):
        """Test clip buffer times are 120 seconds."""
        self.assertEqual(self.detector.CLIP_BEFORE_SECONDS, 120)
        self.assertEqual(self.detector.CLIP_AFTER_SECONDS, 120)


if __name__ == '__main__':
    unittest.main()
