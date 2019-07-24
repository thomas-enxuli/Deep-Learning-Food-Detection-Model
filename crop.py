# coding: utf-8
import cv2
import pandas as pd
import os
def crop_and_resize(train_val_test):
    path = './balanced_resized_images/'+train_val_test
    list_labels = os.listdir(path)
    for label in list_labels:
        csv_path = path+'/'+label+'/'+label+'_data.csv'
        df = pd.read_csv(csv_path)
        for idx in range(len(df.Image_id)):
            image_id = df.Image_id[idx]
            x_min = (int)(df.x_min[idx]*640)
            x_max = (int)(df.x_max[idx]*640)
            y_min = (int)(df.y_min[idx]*640)
            y_max = (int)(df.y_max[idx]*640)
            old_img_path = path+'/'+label+'/'+image_id+'.jpg'
            im = cv2.imread(old_img_path)
            cropped = im[y_min:y_max,x_min:x_max] 
            resized = cv2.resize(cropped,(300,300))
            new_path = './cropped_images/'+train_val_test+'/'+label+'/'+image_id+'.jpg'
            cv2.imwrite(new_path,resized)
            print ('Processed '+train_val_test+' '+label)
