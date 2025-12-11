#!/usr/bin/env python3
"""
Albion Online Combat Clip Extractor CLI
Extracts combat highlights from 1080p/60fps gameplay videos.
"""
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skill_slot_classifier import SkillSlotClassifier
from src.combat_detector import CombatDetector
from src.clip_extractor import ClipExtractor
from src.video_processor import VideoProcessor


class AlbionClipExtractor:
    """Main application for extracting Albion combat clips."""
    
    UI_ERROR_LOG = 'ui_error_videos.txt'
    NO_COMBAT_LOG = 'no_combat_videos.txt'
    
    def __init__(self, model_path=None):
        """Initialize the clip extractor.
        
        Args:
            model_path: path to trained CNN model (optional)
        """
        self.classifier = SkillSlotClassifier(model_path)
        self.combat_detector = CombatDetector(self.classifier)
        self.clip_extractor = ClipExtractor()
    
    def validate_video(self, video_path: str) -> bool:
        """Validate video format and resolution.
        
        Args:
            video_path: path to video file
            
        Returns:
            bool: True if valid
        """
        try:
            video_info = VideoProcessor.get_video_info(video_path)
            
            # Check if video is 1080p
            if video_info['width'] != 1920 or video_info['height'] != 1080:
                self.log_ui_error(video_path, 
                                 f"Invalid resolution: {video_info['width']}x{video_info['height']} (expected 1920x1080)")
                return False
            
            # Check if video has reasonable fps (should be 60fps)
            if video_info['fps'] < 30:
                self.log_ui_error(video_path, 
                                 f"Low FPS: {video_info['fps']} (expected 60)")
                return False
            
            return True
            
        except Exception as e:
            self.log_ui_error(video_path, f"Error reading video: {str(e)}")
            return False
    
    def log_ui_error(self, video_path: str, error_msg: str):
        """Log UI/format errors to ui_error_videos.txt.
        
        Args:
            video_path: path to video file
            error_msg: error message
        """
        with open(self.UI_ERROR_LOG, 'a') as f:
            f.write(f"{video_path}: {error_msg}\n")
    
    def log_no_combat(self, video_path: str):
        """Log videos with no combat to no_combat_videos.txt.
        
        Args:
            video_path: path to video file
        """
        with open(self.NO_COMBAT_LOG, 'a') as f:
            f.write(f"{video_path}\n")
    
    def process_video(self, video_path: str, output_dir: str) -> bool:
        """Process a single video to extract combat clips.
        
        Args:
            video_path: path to input video
            output_dir: directory for output files
            
        Returns:
            bool: True if successful
        """
        print(f"Processing: {video_path}")
        
        # Validate video
        if not self.validate_video(video_path):
            print(f"  ✗ Video validation failed")
            return False
        
        # Detect combat segments
        print(f"  Detecting combat segments...")
        try:
            segments = self.combat_detector.detect_combat_segments(video_path)
        except Exception as e:
            self.log_ui_error(video_path, f"Error detecting combat: {str(e)}")
            print(f"  ✗ Error: {str(e)}")
            return False
        
        if not segments:
            print(f"  ✗ No combat detected")
            self.log_no_combat(video_path)
            return False
        
        print(f"  ✓ Found {len(segments)} combat segment(s)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        video_name = Path(video_path).stem
        output_video = os.path.join(output_dir, f"{video_name}_highlights.mp4")
        output_json = os.path.join(output_dir, f"{video_name}_metadata.json")
        
        # Extract and merge clips
        print(f"  Extracting clips...")
        try:
            success = self.clip_extractor.extract_and_merge_segments(
                video_path, segments, output_video, output_json
            )
            
            if success:
                print(f"  ✓ Output saved to: {output_video}")
                print(f"  ✓ Metadata saved to: {output_json}")
                return True
            else:
                print(f"  ✗ Failed to extract clips")
                return False
                
        except Exception as e:
            self.log_ui_error(video_path, f"Error extracting clips: {str(e)}")
            print(f"  ✗ Error: {str(e)}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract combat highlights from Albion Online gameplay videos'
    )
    parser.add_argument(
        'input',
        help='Input video file or directory containing videos'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory for highlights (default: output)'
    )
    parser.add_argument(
        '-m', '--model',
        help='Path to trained CNN model (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AlbionClipExtractor(model_path=args.model)
    
    # Get list of video files to process
    input_path = Path(args.input)
    
    if input_path.is_file():
        video_files = [str(input_path)]
    elif input_path.is_dir():
        # Find all video files in directory
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
            video_files.extend([str(p) for p in input_path.glob(ext)])
        
        if not video_files:
            print(f"No video files found in {input_path}")
            return 1
    else:
        print(f"Error: {input_path} not found")
        return 1
    
    # Process videos
    print(f"Found {len(video_files)} video(s) to process\n")
    
    success_count = 0
    for video_file in video_files:
        if extractor.process_video(video_file, args.output):
            success_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"Processed {len(video_files)} video(s)")
    print(f"Successfully extracted highlights from {success_count} video(s)")
    print(f"Check {extractor.UI_ERROR_LOG} for error details")
    print(f"Check {extractor.NO_COMBAT_LOG} for videos without combat")
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
