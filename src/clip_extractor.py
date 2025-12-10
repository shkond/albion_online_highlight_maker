"""
Clip extraction and merging for Albion combat highlights.
"""
import cv2
import json
import os
from typing import List, Dict


class ClipExtractor:
    """Extracts and merges combat clips from videos."""
    
    def __init__(self):
        pass
    
    def extract_clip(self, video_path: str, start_frame: int, end_frame: int, 
                     output_path: str) -> bool:
        """Extract a clip from a video.
        
        Args:
            video_path: input video path
            start_frame: starting frame number
            end_frame: ending frame number
            output_path: output video path
            
        Returns:
            bool: True if successful
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return True
    
    def merge_clips(self, clip_paths: List[str], output_path: str) -> bool:
        """Merge multiple clips into one video.
        
        Args:
            clip_paths: list of clip file paths
            output_path: output merged video path
            
        Returns:
            bool: True if successful
        """
        if not clip_paths:
            return False
        
        # Get video properties from first clip
        cap = cv2.VideoCapture(clip_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Merge all clips
        for clip_path in clip_paths:
            cap = cv2.VideoCapture(clip_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
        
        out.release()
        
        return True
    
    def extract_and_merge_segments(self, video_path: str, segments: List[Dict],
                                   output_video_path: str, output_json_path: str,
                                   temp_dir: str = '/tmp/albion_clips') -> bool:
        """Extract combat segments and merge them into one video with metadata.
        
        Args:
            video_path: input video path
            segments: list of combat segment dictionaries
            output_video_path: output merged video path
            output_json_path: output JSON metadata path
            temp_dir: temporary directory for individual clips
            
        Returns:
            bool: True if successful
        """
        if not segments:
            return False
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract individual clips
        clip_paths = []
        for i, segment in enumerate(segments):
            clip_path = os.path.join(temp_dir, f'clip_{i:03d}.mp4')
            success = self.extract_clip(
                video_path,
                segment['clip_start_frame'],
                segment['clip_end_frame'],
                clip_path
            )
            if success:
                clip_paths.append(clip_path)
        
        # Merge clips
        if clip_paths:
            self.merge_clips(clip_paths, output_video_path)
            
            # Create metadata JSON
            metadata = {
                'source_video': os.path.basename(video_path),
                'segments': segments,
                'total_segments': len(segments),
                'output_video': os.path.basename(output_video_path)
            }
            
            with open(output_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Clean up temp clips
            for clip_path in clip_paths:
                try:
                    os.remove(clip_path)
                except:
                    pass
            
            return True
        
        return False
