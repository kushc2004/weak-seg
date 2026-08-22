"""WeakSeg: annotation-efficient semantic segmentation with image-level supervision."""

__version__ = "0.1.0"

# VOC2012 constants shared across the pipeline.
VOC_NUM_CLASSES = 21  # 20 foreground classes + background
VOC_NUM_FG_CLASSES = 20
VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
