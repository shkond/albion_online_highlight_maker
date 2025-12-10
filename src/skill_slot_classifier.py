"""
Lightweight PyTorch CNN for classifying Albion skill slots.
Classes: Normal, Cooldown, Empty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkillSlotCNN(nn.Module):
    """Lightweight CNN for skill slot classification."""
    
    def __init__(self, num_classes=3):
        super(SkillSlotCNN, self).__init__()
        # Input: 3x64x64 (RGB image of skill slot)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64x8x8
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SkillSlotClassifier:
    """Wrapper for skill slot classification."""
    
    SLOT_SIZE = (64, 64)
    CLASSES = ['Normal', 'Cooldown', 'Empty']
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SkillSlotCNN(num_classes=len(self.CLASSES))
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_slot(self, slot_image):
        """Preprocess slot image for model input.
        
        Args:
            slot_image: numpy array (H, W, 3) BGR format
            
        Returns:
            torch.Tensor: preprocessed image (1, 3, 64, 64)
        """
        # Convert BGR to RGB
        slot_rgb = slot_image[:, :, ::-1]
        
        # Normalize to [0, 1]
        slot_normalized = slot_rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        slot_tensor = torch.from_numpy(slot_normalized.transpose(2, 0, 1))
        
        # Add batch dimension
        slot_tensor = slot_tensor.unsqueeze(0)
        
        return slot_tensor.to(self.device)
    
    def classify_slot(self, slot_image):
        """Classify a single skill slot.
        
        Args:
            slot_image: numpy array (H, W, 3) BGR format
            
        Returns:
            str: predicted class ('Normal', 'Cooldown', or 'Empty')
        """
        with torch.no_grad():
            slot_tensor = self.preprocess_slot(slot_image)
            output = self.model(slot_tensor)
            _, predicted = torch.max(output, 1)
            return self.CLASSES[predicted.item()]
    
    def classify_slots(self, slot_images):
        """Classify multiple skill slots.
        
        Args:
            slot_images: list of numpy arrays (H, W, 3) BGR format
            
        Returns:
            list: predicted classes for each slot
        """
        if not slot_images:
            return []
        
        with torch.no_grad():
            # Batch process
            batch = torch.cat([self.preprocess_slot(img) for img in slot_images], dim=0)
            output = self.model(batch)
            _, predicted = torch.max(output, 1)
            return [self.CLASSES[idx] for idx in predicted.cpu().numpy()]
