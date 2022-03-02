# za_emotion_detection

## Introduction

This project aims to identify human emotions using cameras. This work done during deep learning internship at [ZummitAfrica](https://www.linkedin.com/company/zummit-africa/?originalSubdomain=ng). We made use of the [Ferplus](https://github.com/microsoft/FERPlus) dataset. Transfer learning with various neural network architectures were used for classification. They include, Vgg16,EfficientNet0 and MobileNetV2.
Reults are as shown below.

![](https://github.com/jideilori/za_emotion_detection/blob/main/docs/example.gif)


[Video source](https://youtu.be/Zy1h49_L8ME)

At the end it saves the data as a time chart which showsthe emotions detected throughout the video to help management easily locate points of interest such as when the customer was annoyed or happy.

<img src="https://github.com/jideilori/za_emotion_detection/blob/main/docs/emo_vs_time_happy.jpg" width="580">


Other possible use cases of emotion detection include
 - locating interesting areas(funny or shocking) in a video that many people are watching such as in a cinema.

other results can be found [here](https://drive.google.com/drive/folders/1U7H__zWtVjljDRMbpT8n3Z7oL2tJ3bpZ?usp=sharing)

## Results

| Architecture  |Test Accuracy| Validation accuracy  |  Epochs  | Model              |
| --------------| ----------- |--------------------- | ---------| -------------------|
| Vgg16         | 80%         |82%                   | 20       |**[vgg_16.tflite](https://drive.google.com/file/d/1XBm9TxpTwj-XbSB7yddRbaq7rMg0MhMO/view?usp=sharing)**
| EfficientNet0 |             |                      |          |**[EFF](https://drivelink)**
| MobileNetV2   |74%          |76%                   |    50    |**[MobileNetV2.tflite](https://drive.google.com/file/d/18V3LikH5-aVWo8ToV5_lpTHHBDTys6YE/view?usp=sharing)**




## Conclusion
Human expression of emotions is not normally long lived. someone may smile or be shocked for just a few seconds but capturing this emotions can really help. However, getting a more robust dataset inorder to improve the model will be quite difficult. Hence, further study will involve combining several open source datasets such as [EMOTIC](http://sunai.uoc.edu/emotic/index.html) and [AffectNet](https://paperswithcode.com/dataset/affectnet) in order to improve the model.
