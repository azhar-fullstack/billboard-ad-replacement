# Billboard dataset (YOLO format)

Put labeled data here (not committed — see `.gitignore`):

```
datasets/billboard/
  train/images/
  train/labels/
  valid/images/    # or val/images
  valid/labels/
  test/images/     # optional
  test/labels/
```

Then train:

```bash
python src/train.py --data datasets/billboard --epochs 100 --device 0
```

Best weights are copied to `weights/best.pt` for the replace demo.
