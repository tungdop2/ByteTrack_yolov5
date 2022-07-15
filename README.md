# ByteTrack_yolov5
## This is an implement of ByteTrack with YOLOv5 backbone instead of YOLOX.

All code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack) and [YOLOv5](https://github.com/ultralytics/yolov5).

## Usage
1. Install packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Download [pretrain model](num_classes) from yolov5 model zoo.
2. Run the code:
    ```bash
    %cd yolov5
    python demo_track.py 
        <demo type, eg. image, video and webcam>
        --path # path to images or video
        --device # gpu or cpu
        --num_classes
        -c,--ckpt # path to checkpoint
        --tsize # input size, only (608, 1088) for yolov5s,m,n and (800, 1440) for yolov5l,x
    ```
    example: 
    ```bash
    python demo_track.py image --path ./data --device gpu --num_classes 80 --ckpt ./weights/yolov5s.pth --tsize (608, 1088)
    ```


