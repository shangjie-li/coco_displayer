# coco_displayer

A Python tool for displaying COCO dataset

## Requirements
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/coco_displayer.git
   ```
 - Install PyTorch environment with Anaconda
   ```
   conda create -n yolov5.v5.0 python=3.8
   conda activate yolov5.v5.0
   cd coco_displayer
   pip install -r requirements.txt
   ```

## Usages
 - Display data in COCO dataset
   ```
   python coco_displayer.py --images_root=images --info_file=instances_train2017.json
   ```
