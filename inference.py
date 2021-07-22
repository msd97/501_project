import cv2 
import time
import numpy as np
from keras.models import load_model
from keras.backend import argmax
import firebase_admin
from firebase_admin import db

default_app = firebase_admin.initialize_app(options={'databaseURL':"https://artcompanion-8adad-default-rtdb.firebaseio.com/"})
ref = db.reference("/others/")

classes = ["Albrecht Durer", 
    "Gustave Dore", 
    "Ivan Aivazovsky", 
    "Pierre-Auguste Renoir", 
    "Rembrandt", 
    "William Merritt Chase", 
    "Zinaida Serebriakova"
]

classes_fb = ["Albrecht Dürer", 
    "Gustave Doré", 
    "Ivan Aivazovsky", 
    "Pierre-Auguste Renoir", 
    "Rembrandt", 
    "William Merritt Chase", 
    "Zinaida Serebriakova"
]

m_path = './weights/weights_ResNet18_drop_200.hdf5'
model = load_model(m_path)

def predict(frame):

    resized = cv2.resize(frame, (64,64), interpolation=cv2.INTER_LINEAR)
    resized = np.array(resized)
    resized = np.expand_dims(resized, axis=0)
    pred = model.predict(resized, verbose=1)
    res = int(argmax(pred[0]))
    pred_class = classes[res]
    ref.update({"detection":classes_fb[res]})
    conf_level = pred[0][res] * 100

    return pred_class, conf_level

def gstreamer_pipeline(
    sensor_id=0,
    #sensor_mode=3,
    capture_width=3280,
    capture_height=2464,
    display_width=816,
    display_height=616,
    framerate=21/1,
    flip_method=2,
):
    return (
       "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
        % (
            sensor_id,
            #sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('demo_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#cap = cv2.VideoCapture(0)

mark = "No detection yet"
init_time = time.time()
curr_time = time.time()

while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if curr_time - init_time > 0.5:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, conf_level = predict(frame)
        mark = label + " " + "{:.2f}%".format(conf_level)
        init_time = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.putText(frame, mark, (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    out.write(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    curr_time = time.time()
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()