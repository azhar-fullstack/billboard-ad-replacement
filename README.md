# Billboard Detection & Ad Replacement

Detect billboards in images/video with YOLO, then warp a new ad onto the board with OpenCV.

```
Train YOLO → detect/segment billboard → warp new ad → blend → output
```

---

## Structure

```
billboard-ad-replacement/
├── assets/                 # sample billboard + ad (+ before/after preview)
├── datasets/billboard/     # your YOLO train/val data (local)
├── demos/                  # generated outputs (local)
├── scripts/
│   └── demo_image.py       # one-command portfolio demo
├── src/
│   ├── train.py            # train billboard YOLO detector
│   ├── replace_image.py    # image + simple video replacement
│   └── replace_video.py    # stable video engine (seg + Kalman/EMA)
├── weights/                # best.pt (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## 1) Train

Prepare YOLO folders under `datasets/billboard/` (see that folder’s README), then:

```bash
python src/train.py --data datasets/billboard --epochs 100 --device 0
```

Copies `best.pt` → `weights/best.pt`.

Or pass an existing yaml:

```bash
python src/train.py --data path/to/data.yaml --model yolo11n.pt
```

---

## 2) Replace (demo)

```bash
python scripts/demo_image.py --device cpu
```

Writes `demos/replaced.jpg` and before/after collage.

Or:

```bash
python src/replace_image.py image --device cpu
```

---

## 3) Video (advanced)

```bash
python src/replace_video.py --video-in path/to/input.mp4 --ad assets/sample_ad.webp --out demos/replaced_video.mp4
```

See `python src/replace_video.py -h` for Kalman/EMA, mosaic panels, occluders, etc.

---

## Stack

Python · Ultralytics YOLO · OpenCV · NumPy · PyTorch

Weights, long videos, and training images/labels are excluded from the repo.

**Author:** Azhar Mehmood
