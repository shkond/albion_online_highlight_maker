"""
Unit tests for SkillSlotClassifier
"""
import unittest
import numpy as np
import torch
from src.skill_slot_classifier import SkillSlotClassifier, SkillSlotCNN


class TestSkillSlotCNN(unittest.TestCase):
    """Test the CNN model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = SkillSlotCNN(num_classes=3)
        self.assertIsNotNone(model)
    
    def test_model_forward(self):
        """Test forward pass."""
        model = SkillSlotCNN(num_classes=3)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 3))


class TestSkillSlotClassifier(unittest.TestCase):
    """Test the classifier wrapper."""
    
    def setUp(self):
        """Set up test classifier."""
        self.classifier = SkillSlotClassifier()
    
    def test_classifier_creation(self):
        """Test classifier can be created."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(len(self.classifier.CLASSES), 3)
    
    def test_preprocess_slot(self):
        """Test slot preprocessing."""
        # Create dummy slot image (BGR)
        slot_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Preprocess
        slot_tensor = self.classifier.preprocess_slot(slot_image)
        
        # Check shape
        self.assertEqual(slot_tensor.shape, (1, 3, 64, 64))
        
        # Check value range [0, 1]
        self.assertTrue(torch.all(slot_tensor >= 0))
        self.assertTrue(torch.all(slot_tensor <= 1))
    
    def test_classify_slot(self):
        """Test single slot classification."""
        # Create dummy slot image
        slot_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Classify
        result = self.classifier.classify_slot(slot_image)
        
        # Check result is valid class
        self.assertIn(result, self.classifier.CLASSES)
    
    def test_classify_slots(self):
        """Test batch slot classification."""
        # Create dummy slot images
        slot_images = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        # Classify
        results = self.classifier.classify_slots(slot_images)
        
        # Check results
        self.assertEqual(len(results), 8)
        for result in results:
            self.assertIn(result, self.classifier.CLASSES)
    
    def test_classify_empty_list(self):
        """Test classifying empty list."""
        results = self.classifier.classify_slots([])
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()
