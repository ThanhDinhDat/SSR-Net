import os
import cv2
import dlib
import numpy as np
import argparse
from SSRNET_model import SSR_net, SSR_net_general
import sys
import timeit
import time
from moviepy.editor import *
from mtcnn.mtcnn import MTCNN
from keras import backend as K
import tensorflow as tf
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot):
    
    #for i, d in enumerate(detected):
    for i, d in enumerate(detected):
        #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        x,y,w,h = d['box']
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)
        
        faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        
        faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)
        
    
    start_time = timeit.default_timer()
    if len(detected) > 0:
        # predict ages and genders of the detected faces
        predicted_ages = model.predict(faces)
        predicted_genders = model_gender.predict(faces)
        

    # draw results
    for i, d in enumerate(detected):
        #label = "{}~{}, {}".format(int(predicted_ages[i]*4.54),int((predicted_ages[i]+1)*4.54),
        #                       "F" if predicted_genders[i][0] > 0.5 else "M")
        x,y,w,h = d['box']
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        gender_str = 'male'
        if predicted_genders[i]<0.5:
            gender_str = 'female'
        label = "{},{}".format(int(predicted_ages[i]),gender_str)
        
        draw_label(input_img, (x1, y1), label)
    
    elapsed_time = timeit.default_timer()-start_time
    time_network = time_network + elapsed_time
    
    
    
    start_time = timeit.default_timer()

    #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("result", input_img)
        
    
    elapsed_time = timeit.default_timer()-start_time
    time_plot = time_plot + elapsed_time

    return input_img,time_network,time_plot

def main():
    K.set_learning_phase(0) # make sure its testing mode
    weight_file = "../pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "../pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    # for face detection
    # detector = dlib.get_frontal_face_detector()
    detector = MTCNN()
    try:
        os.mkdir('./img')
    except OSError:
        pass
    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)

    model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(weight_file_gender)
    clip = VideoFileClip(sys.argv[1]) # can be gif or movie
    
    #python version
    pyFlag = ''
    if len(sys.argv)<3 :
        pyFlag = '2' #default to use moviepy to show, this can work on python2.7 and python3.5
    elif len(sys.argv)==3:
        pyFlag = sys.argv[2] #python version
    else:
        print('Wrong input!')
        sys.exit()

    img_idx = 0
    detected = '' #make this not local variable 
    time_detection = 0
    time_network = 0
    time_plot = 0
    ad = 0.4
    skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
    for img in clip.iter_frames():
        img_idx = img_idx + 1
        
        
        input_img = img #using python2.7 with moivepy to show th image without channel flip
        
        if pyFlag == '3':
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_h, img_w, _ = np.shape(input_img)
        input_img = cv2.resize(input_img, (1024, int(1024*img_h/img_w)))
        img_h, img_w, _ = np.shape(input_img)
        
        
        
        if img_idx==1 or img_idx%skip_frame == 0:
            
            # detect faces using dlib detector
            start_time = timeit.default_timer()
            detected = detector.detect_faces(input_img)
            elapsed_time = timeit.default_timer()-start_time
            time_detection = time_detection + elapsed_time
            faces = np.empty((len(detected), img_size, img_size, 3))
            input_img,time_network,time_plot = draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot)
        
        #Show the time cost (fps)
        # print('avefps_time_detection:',img_idx/time_detection)
        # print('avefps_time_network:',img_idx/time_network)
        # print('avefps_time_plot:',img_idx/time_plot)
        # print('===============================')
        if pyFlag == '3':
            key = cv2.waitKey()
            if key == 27:
                break


if __name__ == '__main__':
    main()
