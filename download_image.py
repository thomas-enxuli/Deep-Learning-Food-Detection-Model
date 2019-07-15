#google image data info
# use the following link
#https://github.com/widemeadows/openimages-dataset


import pandas as pd
import os.path
import os
import urllib.request
from resize import *

new_size = 640


def helper_extract_image_id(train_validation_test,extracted_class):
    extracted = []
    image_id_data = pd.read_csv(train_validation_test+"-annotations-bbox.csv",usecols=['ImageID','LabelName','XMin','XMax','YMin','YMax'])
    img_cnt = 0

    for i in range(len(image_id_data.ImageID)):
        if image_id_data.LabelName[i] in extracted_class['Class_ID'].values:
            idx_in_data = extracted_class.Class_ID[extracted_class.Class_ID==image_id_data.LabelName[i]].index.tolist()[0]
            extracted.append([
                image_id_data.ImageID[i],
                extracted_class.Class_ID[idx_in_data],
                extracted_class.Class_Description[idx_in_data],
                image_id_data.XMin[i],
                image_id_data.XMax[i],
                image_id_data.YMin[i],
                image_id_data.YMax[i]
                ])
            img_cnt += 1

    return [extracted,img_cnt]


def image_id_csv():

    #first import class-descriptions.csv
    extracted_class = pd.read_csv('class-description.csv')
    # training

    [extracted_image_id_train,img_cnt_train] = helper_extract_image_id('train',extracted_class)
    print (str(img_cnt_train)+' training images extracted.')

    export_df_train = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_train],
                'Class_id':[i[1] for i in extracted_image_id_train],
                'class_description':[i[2] for i in extracted_image_id_train],
                'x_min':[i[3] for i in extracted_image_id_train],
                'x_max':[i[4] for i in extracted_image_id_train],
                'y_min':[i[5] for i in extracted_image_id_train],
                'y_max':[i[6] for i in extracted_image_id_train]
                })
    export_csv_train = export_df_train.to_csv ('image_id+class_id_train.csv', index = None, header=True)
    print ('(training data) Finished exporting to image_id+class_id_train.csv')

    #validation
    [extracted_image_id_validation,img_cnt_validation] = helper_extract_image_id('validation',extracted_class)
    print (str(img_cnt_validation)+' validation images extracted.')

    export_df_validation = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_validation],
                'Class_id':[i[1] for i in extracted_image_id_validation],
                'class_description':[i[2] for i in extracted_image_id_validation],
                'x_min':[i[3] for i in extracted_image_id_validation],
                'x_max':[i[4] for i in extracted_image_id_validation],
                'y_min':[i[5] for i in extracted_image_id_validation],
                'y_max':[i[6] for i in extracted_image_id_validation]
                })
    export_csv_validation = export_df_validation.to_csv ('image_id+class_id_validation.csv', index = None, header=True)
    print ('(validation data) Finished exporting to image_id+class_id_validation.csv')

    #test
    [extracted_image_id_test,img_cnt_test] = helper_extract_image_id('test',extracted_class)
    print (str(img_cnt_test)+' testing images extracted.')

    export_df_test = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_test],
                'Class_id':[i[1] for i in extracted_image_id_test],
                'class_description':[i[2] for i in extracted_image_id_test],
                'x_min':[i[3] for i in extracted_image_id_test],
                'x_max':[i[4] for i in extracted_image_id_test],
                'y_min':[i[5] for i in extracted_image_id_test],
                'y_max':[i[6] for i in extracted_image_id_test]
                })
    export_csv_test = export_df_test.to_csv ('image_id+class_id_test.csv', index = None, header=True)
    print ('(testing data) Finished exporting to image_id+class_id_test.csv')


def helper_search(image_url_data,class_id_data):
    extracted_image_url = []
    img_cnt = 0
    for i in range(len(image_url_data.ImageID)):
        if (image_url_data.ImageID[i] in class_id_data['Image_id'].values):
            idx_in_data = class_id_data.Image_id[class_id_data.Image_id==image_url_data.ImageID[i]].index.tolist()[0]
            extracted_image_url.append([
                image_url_data.ImageID[i],
                image_url_data.Title[i],
                image_url_data.Thumbnail300KURL[i],
                class_id_data.Class_id[idx_in_data],
                class_id_data.class_description[idx_in_data],
                class_id_data.x_min[idx_in_data],
                class_id_data.x_max[idx_in_data],
                class_id_data.y_min[idx_in_data],
                class_id_data.y_max[idx_in_data]
                ])
            img_cnt += 1

    return [img_cnt,extracted_image_url]

def helper_export_df(extracted_image_url,test_val_test):
    export_df= pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_url],
                'Image_title':[i[1] for i in extracted_image_url],
                'Image_url':[i[2] for i in extracted_image_url],
                'Class_id':[i[3] for i in extracted_image_url],
                'class_description':[i[4] for i in extracted_image_url],
                'x_min':[i[5] for i in extracted_image_url],
                'x_max':[i[6] for i in extracted_image_url],
                'y_min':[i[7] for i in extracted_image_url],
                'y_max':[i[8] for i in extracted_image_url]
                })
    export_csv = export_df.to_csv ('image_url_extracted_'+test_val_test+'.csv', index = None, header=True)

def image_url():

############################################################
    #train
    #if not os.path.isfile('./extracted_info/image_url_extracted_train.csv'):
        #read url file
    # image_url_data = pd.read_csv('train-images-boxable-with-rotation.csv',usecols=['ImageID','Title','Thumbnail300KURL'])
    #
    # #read class id file
    # class_id_data = pd.read_csv('image_id+class_id_train.csv')
    #
    # [train_img_cnt,extracted_image_url_train] = helper_search(image_url_data,class_id_data)
    #
    # print (str(train_img_cnt)+' train images url extracted.')
    #
    # helper_export_df(extracted_image_url_train,'train')
    # print ('(train data) Finished exporting to image_url_extracted_train.csv')

    # else:
    #     print ('(train data) Already exported to image_url_extracted_train.csv')



########################################################################################
    #validaiton
    image_url_data = pd.read_csv('validation-images-with-rotation.csv',usecols=['ImageID','Title','Thumbnail300KURL'])

    #read class id file
    class_id_data = pd.read_csv('image_id+class_id_validation.csv')

    [validation_img_cnt,extracted_image_url_validation] = helper_search(image_url_data,class_id_data)

    print (str(validation_img_cnt)+' validation images url extracted.')

    helper_export_df(extracted_image_url_validation,'validation')
    print ('(validation data) Finished exporting to image_url_extracted_validation.csv')


    image_url_data = pd.read_csv('test-images-with-rotation.csv',usecols=['ImageID','Title','Thumbnail300KURL'])

    #read class id file
    class_id_data = pd.read_csv('image_id+class_id_test.csv')

    [test_img_cnt,extracted_image_url_test] = helper_search(image_url_data,class_id_data)

    print (str(test_img_cnt)+' test images url extracted.')

    helper_export_df(extracted_image_url_test,'test')
    print ('(test data) Finished exporting to image_url_extracted_test.csv')

def sort_csv_with_class():
    image_url_data = pd.read_csv('image_url_extracted_train.csv')
    export_sorted_csv_train = image_url_data.sort_values(by=['Class_id']).to_csv ('image_url_extracted_train.csv', index = None, header=True)

    image_url_data = pd.read_csv('image_url_extracted_validation.csv')
    export_sorted_csv_validation = image_url_data.sort_values(by=['Class_id']).to_csv ('image_url_extracted_validation.csv', index = None, header=True)

    image_url_data = pd.read_csv('image_url_extracted_test.csv')
    export_sorted_csv_test = image_url_data.sort_values(by=['Class_id']).to_csv ('image_url_extracted_test.csv', index = None, header=True)

def download(train_validation_test):
    image_url_data = pd.read_csv('./new_resources/image_url_extracted_'+train_validation_test+'.csv')
    for i in range(len(image_url_data.Image_url)):
        name = "./images/"+train_validation_test+"/"+image_url_data.class_description[i]+"/"+image_url_data.Image_id[i]
        url = image_url_data.Image_url[i]
        print ('Image '+str(i).zfill(6) + '///' + image_url_data.Image_id[i])

        try:
            print ('Downloading from '+url)
            print ('Saving to '+name)
            urllib.request.urlretrieve(url,name)
            print ('Success ...\n\n')
        except:
            print ('Failed ...\n\n')
    print ('Finished downloading '+train_validation_test+' images ...')

def resize_all():
    grandmother = os.listdir('./images') # train, test, val
    for i in grandmother:
        parent = os.listdir('./images/'+i) # bacon, ...
        for j in parent:
            child = os.listdir('./images/'+i+'/'+j) # images
            idx = 0
            for k in child:
                old_path = './images/'+i+'/'+j+'/'+k
                print ('old file path: '+old_path)
                new_path = './resized_images/'+i+'/'+j+'/'+k+'.jpg'
                idx += 1
                resize_im(old_path,new_path)
    print ('Finished resizing all images ...')

def change_box_resize(train_validation_test):

    old_data = pd.read_csv('./new_resources/image_url_extracted_'+train_validation_test+'.csv')
    parent = os.listdir('./images/'+train_validation_test)
    for i in parent:
        child = os.listdir('./images/'+train_validation_test+'/'+i)
        new_data = []
        for j in child:
            im = cv2.imread('./images/'+train_validation_test+'/'+i+'/'+j)
            y = im.shape[0]
            x = im.shape[1]
            idx_in_data = old_data.Image_id[old_data.Image_id==j].index.tolist()[0]
            new_x_min = ((new_size - x)/2.0 + (old_data.x_min[idx_in_data]*x))/new_size
            new_x_max = ((new_size - x)/2.0 + (old_data.x_max[idx_in_data]*x))/new_size
            new_y_min = ((new_size - y)/2.0 + (old_data.y_min[idx_in_data]*y))/new_size
            new_y_max = ((new_size - y)/2.0 + (old_data.y_max[idx_in_data]*y))/new_size

            new_data.append([
                old_data.Image_id[idx_in_data],
                old_data.Image_title[idx_in_data],
                old_data.Image_url[idx_in_data],
                old_data.Class_id[idx_in_data],
                old_data.class_description[idx_in_data],
                new_x_min,
                new_x_max,
                new_y_min,
                new_y_max
                ])
        export_df= pd.DataFrame({
                    'Image_id':[k[0] for k in new_data],
                    'Image_title':[k[1] for k in new_data],
                    'Image_url':[k[2] for k in new_data],
                    'Class_id':[k[3] for k in new_data],
                    'class_description':[k[4] for k in new_data],
                    'x_min':[k[5] for k in new_data],
                    'x_max':[k[6] for k in new_data],
                    'y_min':[k[7] for k in new_data],
                    'y_max':[k[8] for k in new_data]
                    })
        #export_csv = export_df.to_csv ('./images/'+train_validation_test+'/'+i+'/'+i+'_data.csv', index = None, header=True)
        export_csv = export_df.to_csv ('./resized_images/'+train_validation_test+'/'+i+'/'+i+'_data.csv', index = None, header=True)
        print ('Finished '+i)

def make_folders():
    os.mkdir('./resized_images/')
    print ('Made folder ./resized_images/')
    os.mkdir('./resized_images/train/')

    os.mkdir('./resized_images/validation/')

    os.mkdir('./resized_images/test/')
    class_info = pd.read_csv('class-description.csv')
    for i in range(len(class_info.Class_ID)):
        os.mkdir('./resized_images/train/'+class_info.Class_Description[i])
    print ('Made folder ./resized_images/train/')
    for i in range(len(class_info.Class_ID)):
        os.mkdir('./resized_images/validation/'+class_info.Class_Description[i])
    print ('Made folder ./resized_images/validation/')
    for i in range(len(class_info.Class_ID)):
        os.mkdir('./resized_images/test/'+class_info.Class_Description[i])
    print ('Made folder ./resized_images/test/')

##############################################################
#main

#image_id_csv()
#image_url()
#sort_csv_with_class()



#check if the first and second step is finished
# if os.path.isfile('./extracted_info/image_id+class_id_test.csv') and os.path.isfile('./extracted_info/image_id+class_id_train.csv') and os.path.isfile('./extracted_info/image_id+class_id_validation.csv'):
#     print ('First and second step finished ... ')
#     [food_cnt,extracted_class_id,sel_class] = get_class_id()
#     print ('Extracting URL ...')
#     image_url()
#     print ('FINISHED EXTRACTING URL  ... ')
#     sort_csv_with_class()
#     print ('SORTED')
# else:
#     print ('Running first and second step ...')
#     image_id_csv()
#     print ('First and second step finished ... ')
#     print ('Extracting URL ...')
#     image_url()
#     print ('FINISHED EXTRACTING URL  ... ')
#     sort_csv_with_class()
#     print ('SORTED')
#
# if not os.path.exists('./resized_data/'):

# print ('Finished making folders')
# resize_all()


#download('validation')
#download('test')
#download('train')
#third import images.csv
#go through the file
# if image_id in the list:
#	save the [image url, image description..., class id, class description]
