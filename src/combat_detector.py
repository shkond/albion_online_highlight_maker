"""
Combat detection logic for Albion Online videos.
"""
from typing import List, Tuple, Dict
import cv2
from src.skill_slot_classifier import SkillSlotClassifier
from src.video_processor import VideoProcessor


class CombatDetector:
    """Detects combat segments in Albion Online videos."""
    
    # Constants from requirements
    MIN_MOUNTED_EMPTY_SLOTS = 4
    COOLDOWN_START_FRAMES = 3
    NO_COOLDOWN_END_SECONDS = 60
    CLIP_BEFORE_SECONDS = 120  # 2 minutes
    CLIP_AFTER_SECONDS = 120   # 2 minutes
    
    def __init__(self, classifier: SkillSlotClassifier):
        self.classifier = classifier
        self.video_processor = VideoProcessor()
    
    def is_mounted(self, slot_classes: List[str]) -> bool:
        """Check if player is mounted (4+ empty slots).
        
        Args:
            slot_classes: list of 8 slot classifications
            
        Returns:
            bool: True if mounted
        """
        empty_count = slot_classes.count('Empty')
        return empty_count >= self.MIN_MOUNTED_EMPTY_SLOTS
    
    def has_cooldown(self, slot_classes: List[str]) -> bool:
        """Check if any slot is on cooldown.
        
        Args:
            slot_classes: list of 8 slot classifications
            
        Returns:
            bool: True if at least one slot is on cooldown
        """
        return 'Cooldown' in slot_classes
    
    def detect_combat_segments(self, video_path: str) -> List[Dict]:
        """Detect combat segments in a video.
        
        Args:
            video_path: path to video file
            
        Returns:
            list: combat segments with start/end frame numbers
        """
        video_info = self.video_processor.get_video_info(video_path)
        fps = video_info['fps']
        
        cap = cv2.VideoCapture(video_path)
        
        segments = []
        current_segment = None
        cooldown_frames = 0  # Consecutive frames with cooldown
        no_cooldown_frames = 0  # Consecutive frames without cooldown
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract and classify skill slots
            slot_images = self.video_processor.extract_skill_slots(frame)
            slot_classes = self.classifier.classify_slots(slot_images)
            
            # Check if mounted (ignore combat detection)
            if self.is_mounted(slot_classes):
                frame_idx += 1
                continue
            
            # Check for cooldown
            has_cooldown = self.has_cooldown(slot_classes)
            
            # Check for red screen (death)
            is_red = self.video_processor.is_red_screen(frame)
            
            # Combat start detection: 1+ Cooldown slot for 3 consecutive frames
            if has_cooldown:
                cooldown_frames += 1
                no_cooldown_frames = 0
                
                if cooldown_frames >= self.COOLDOWN_START_FRAMES and current_segment is None:
                    # Start new combat segment
                    start_frame = max(0, frame_idx - self.COOLDOWN_START_FRAMES + 1)
                    current_segment = {
                        'start_frame': start_frame,
                        'end_frame': None,
                        'death': False
                    }
            else:
                no_cooldown_frames += 1
                cooldown_frames = 0
            
            # Combat end detection
            if current_segment is not None:
                # End condition 1: No cooldown for 60 seconds
                no_cooldown_duration = no_cooldown_frames / fps
                if no_cooldown_duration >= self.NO_COOLDOWN_END_SECONDS:
                    current_segment['end_frame'] = frame_idx - no_cooldown_frames
                    segments.append(current_segment)
                    current_segment = None
                    no_cooldown_frames = 0
                
                # End condition 2: Red screen (death)
                elif is_red:
                    current_segment['end_frame'] = frame_idx
                    current_segment['death'] = True
                    segments.append(current_segment)
                    current_segment = None
                    no_cooldown_frames = 0
            
            frame_idx += 1
        
        cap.release()
        
        # Handle segment that didn't end
        if current_segment is not None:
            current_segment['end_frame'] = frame_idx - 1
            segments.append(current_segment)
        
        # Add buffer times to segments
        for segment in segments:
            segment['clip_start_frame'] = max(0, segment['start_frame'] - int(self.CLIP_BEFORE_SECONDS * fps))
            segment['clip_end_frame'] = min(video_info['frame_count'] - 1, 
                                            segment['end_frame'] + int(self.CLIP_AFTER_SECONDS * fps))
            segment['start_time'] = segment['start_frame'] / fps
            segment['end_time'] = segment['end_frame'] / fps
            segment['clip_start_time'] = segment['clip_start_frame'] / fps
            segment['clip_end_time'] = segment['clip_end_frame'] / fps
        
        return segments
