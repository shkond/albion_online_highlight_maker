# Albion Online Highlight Maker

Python CLI tool to extract combat highlights from Albion Online gameplay videos (1080p/60fps).

## Features

- **Automated Combat Detection**: Uses a lightweight PyTorch CNN to classify skill slots (Normal/Cooldown/Empty)
- **Smart Clip Extraction**: Detects combat start/end with configurable buffers
- **Video Processing**: Merges combat segments into a single MP4 with JSON metadata
- **Error Logging**: Tracks UI errors and videos without combat

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process a single video:
```bash
python src/main.py input_video.mp4 -o output_directory
```

### Process all videos in a directory:
```bash
python src/main.py videos/ -o highlights/
```

### With a trained model:
```bash
python src/main.py input_video.mp4 -m model.pth -o output/
```

## Combat Detection Logic

1. **Skill Slot Classification**: Crops 8 skill slots from 1080p frames and classifies each as:
   - Normal (skill available)
   - Cooldown (skill on cooldown - combat indicator)
   - Empty (no skill equipped)

2. **Mounted Detection**: Ignores frames with 4+ empty slots (player is mounted)

3. **Combat Start**: Detected when 1+ skill slots show cooldown for 3 consecutive frames

4. **Combat End**: Triggered by either:
   - No cooldown detected for 60 seconds
   - Red screen detected (death)

5. **Clip Extraction**: Each combat segment is clipped with:
   - 2 minutes before combat start
   - 2 minutes after combat end

6. **Output**:
   - Single merged MP4 file with all combat segments
   - JSON metadata file with segment information
   - `ui_error_videos.txt` - Videos with format/resolution errors
   - `no_combat_videos.txt` - Videos without detected combat

## Project Structure

```
.
├── src/
│   ├── main.py                    # CLI entry point
│   ├── skill_slot_classifier.py   # PyTorch CNN for slot classification
│   ├── combat_detector.py         # Combat detection logic
│   ├── video_processor.py         # Video processing utilities
│   └── clip_extractor.py          # Clip extraction and merging
├── tests/
│   ├── test_skill_slot_classifier.py
│   ├── test_video_processor.py
│   ├── test_combat_detector.py
│   └── test_clip_extractor.py
└── requirements.txt

```

## Running Tests

```bash
python -m pytest tests/
# or
python -m unittest discover tests
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- PyTorch
- NumPy

## Notes

- Input videos must be 1920x1080 resolution
- Recommended: 60 FPS videos for best detection
- Skill slot positions are calibrated for standard Albion UI layout
- The CNN model can be trained on labeled skill slot images for improved accuracy