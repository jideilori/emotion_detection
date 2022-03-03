import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
interpreter = tf.lite.Interpreter(model_path="emotion.tflite")
interpreter.allocate_tensors()

offset = 0
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
emotions_df = []
col_names = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","Unknown", "NF"]

# Default webcam
cap = cv2.VideoCapture(0)

width,height = 640,480
fps =  cap.get(cv2.CAP_PROP_FPS)
size = (width, height)
vids = cv2.VideoWriter("emotion_results.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

def convToSec(df,f):
    '''
    Takes a mode of f entries and returns a new df with all the modes
    '''
    f_emo = []
    n = 0
    bool_chk = True
    while bool_chk :
        f_emo.append(df[n:n+f].value_counts().index[0][0]) 
        n = n + f 
        if n > len(df):
            f_emo.append(df[n-f:len(df)].value_counts().index[0][0])
            bool_chk = False 
    return pd.DataFrame(f_emo)

def check_emotion(crop_img):
  '''
  Args:
    crop_img (array):
  Returns:
    int: Integer would be between 0-9 representing the 10 
         emotions
  '''
  img = cv2.resize(crop_img,(48,48))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = np.repeat(img[..., np.newaxis], 3, -1)

  cv2.imshow('processed',img)
  img =img / 255
  img = img.astype('float32')
  
  input_shape = input_details[0]['shape']
  input_tensor= np.expand_dims(img,0)
  interpreter.set_tensor(input_details[0]['index'], input_tensor)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])

  pred = np.squeeze(output_data)
  highest_pred_loc = np.argmax(pred)
  return highest_pred_loc

def main():
  '''
  Detects face and outputs an emotion
  '''
  
  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      
      if not success:
        print("Ignoring empty camera frame.")
        break
      image = cv2.resize(image,(640,480))
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          
          location_data = detection.location_data
          bb = location_data.relative_bounding_box
          
          bbox_points = {
              "xmin" : int(bb.xmin * width),
              "ymin" : int(bb.ymin * height),
              "xmax" : int(bb.width * width + bb.xmin * width),
              "ymax" : int(bb.height * height + bb.ymin * height)
          }

          x,y,w,h = bbox_points['xmin'],bbox_points['ymin'],bbox_points['xmax'],bbox_points['ymax']
          try:
            cropped_image = image[y-offset:h+ offset,x-offset:w+offset]
            emo_det = check_emotion(cropped_image)
            emotions_df.append(emo_det)
            cv2.putText(image,col_names[emo_det],(w,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(0,0,255),2)
            cv2.rectangle(image,(x,y),(w,h),(0,255,0),2)
            
            

          except:
            pass
      cv2.imshow('Face', image)
      vids.write(image)
      if cv2.waitKey(30) & 0xFF == 27:
        break
  cap.release()
  vids.release()

  emo_d= pd.DataFrame(emotions_df) 
  emo_10 = convToSec(emo_d, 10)
  emo_10 = emo_10.rename(columns={0:"emotions"})
  plt.figure(figsize=(10,7))
  plt.scatter(emo_10.index, emo_10.emotions)
  plt.yticks(np.arange(0, 9.15, 1),col_names)
  plt.ylabel("Emotions",fontsize=12)
  plt.xlabel("Time(Seconds)",fontsize=12)
  plt.title("Captured emotions over time",fontsize=12)
  plt.savefig("emo_vs_time.jpg")
  plt.show()

  emo_10["emotions"] = emo_10["emotions"].apply(lambda x : col_names[x])
  emo_10.value_counts().plot.barh()
  plt.title("Total time for each emotion",fontsize=12)
  plt.ylabel("Emotions",fontsize=12)
  plt.xlabel("Time(Seconds)",fontsize=12)
  plt.savefig("total_emo.jpg")
  plt.show()

if __name__ == "__main__":
  main()