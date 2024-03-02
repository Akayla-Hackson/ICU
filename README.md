<h1 align="center">IC🫵: Multiple Object Detection in the Domain of Basketball</h1>
<p align="center">
  <img src="input_example.jpg" alt="Image 1" width="400"/>
  <span style="font-size: 200px;">&rarr;</span>
  <img src="prediction_example.jpg" alt="Image 2" width="400"/>
</p>
<h2 align="center">Changing the game one detection at a time! </h2>
<h3 align="center">You are able to test out any of the 4 implemented models. See directions below :) </h3>

## To run on the SportsMOT dataset


### 1. Download data from [https://github.com/MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT)
### 2. Pick a model to run on
- **SSD or Faster RCNN**
     ```bash
     python ./object_detection/eval.py --model <enter model of choice>
     ```
- **Yolov8**
     ```bash
     python ./object_detection/eval_yolov8.py
     ```
    - Note: You may finetune by running:
       ```bash
       python ./object_detection/finetune_yolov8.py
       ```
       Then run the following to use the fine-tuned weights:
       ```bash
       python ./object_detection/eval_yolov8.py --weights <insert weight path>
       ```
- **Detectron2**
     - Run command:
       ```bash
       python ./object_detection/eval_detectron2.py
       ```
     - Note: You may finetune by running:
       ```bash
       python ./object_detection/finetune_detectron2.py
       ```
       Then run the following to use the fine-tuned weights:
       ```bash
       python ./object_detection/eval_detectron2.py --weights <insert weight path>
       ```


   
