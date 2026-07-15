# Billboard Detection & Ad Replacement

Computer-vision pipeline to detect billboards in video/images, track them, and replace ad creatives (YOLO + OpenCV).

## Setup

```bash
cd BillBoardDetection
pip install ultralytics opencv-python pillow numpy
python test_trained_model.py image --input billBoard.webp --output test_image_replaced.jpg
```

Large model weights and long videos are not included — place your own `best.pt` / media locally.
