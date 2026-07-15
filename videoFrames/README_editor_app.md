# Image Coloring Web App

This app lets you:
- load all images from this folder,
- paint red regions,
- erase painted regions,
- save edited outputs into `colored_pic`,
- move previous/next image,
- resume from last image when reopening.

## Run

1. Open terminal in this folder:
   - `D:\Editors\Vs code\Code\03_FiverrProjects\videoFrames`
2. Install dependencies (one time):
   - `pip install flask opencv-python pillow numpy`
3. Start app:
   - `python app.py`
4. Open browser:
   - `http://127.0.0.1:8000`

## Controls

- `R`: paint red
- `E`: erase
- `S`: save current image
- `A`: previous image
- `D`: next image
- `[`: decrease brush size
- `]`: increase brush size

## Output

- Colored outputs: `colored_pic/<same_image_name>`
- Mask files: `colored_pic/_masks/<image_stem>.png`
- Resume state: `.editor_state.json`

Original images are never modified.
