#google image data info
# use the following link
#https://github.com/widemeadows/openimages-dataset


import pandas as pd
import os.path
import os
import urllib.request
from resize import *



def get_class_id():
    sel_class= [] # this list will store extracted extracted [class_id,class_description]
    extracted_class_id = [] # this list is for storing class_id only

    class_description_col_names = ['class_id','class_description']
    description_data = pd.read_csv("./ImageSrc/class-descriptions.csv",names=class_description_col_names)
    #description_data
    #description_data['class_id']
    #description_data

    food_list = pd.read_csv("food_list.txt", delimiter = '\n',names=['food_name'])

    food_cnt = 0

    for i in range(len(food_list.food_name)):
        if (food_list.food_name[i] in description_data.class_description.values):
            idx_in_data = description_data.class_description[description_data.class_description==food_list.food_name[i]].index.tolist()[0]
            extracted_class_id.append(description_data.class_id[idx_in_data])
            sel_class.append([
                description_data.class_id[idx_in_data],
                description_data.class_description[idx_in_data]
                ])

            #print (food_list['food_name'][i])
            food_cnt += 1
    print (str(food_cnt)+' classes of foods extracted.')
    return [food_cnt,extracted_class_id,sel_class]


def image_id_csv():

    extracted_image_id_train = []
    extracted_image_id_validation = []
    extracted_image_id_test = []

    #first import class-descriptions.csv
    #look for classes with food and add [class id, class description] to the list
    [food_cnt,extracted_class_id,sel_class] = get_class_id()



    #second import annotations-human.csv
    #go through the file
    # if class_id in the list:
    #	save the [image id, class id, class description]

    # ImageID Source   LabelName  Confidence
    #image_id_col_names = ['image_id','human','class_id','confidence_level']
    # training
    image_id_data_train = pd.read_csv("./ImageSrc/train/annotations-human.csv")
    train_img_cnt = 0

    for i in range(len(image_id_data_train.ImageID)):
        if (image_id_data_train.LabelName[i] in extracted_class_id) and image_id_data_train.Confidence[i]:
            extracted_image_id_train.append([
                image_id_data_train.ImageID[i],
                sel_class[extracted_class_id.index(image_id_data_train.LabelName[i])][0],
                sel_class[extracted_class_id.index(image_id_data_train.LabelName[i])][1]
                ])
            #print (food_list['food_name'][i])
            train_img_cnt += 1

    print (str(train_img_cnt)+' training images extracted.')

    export_df_train = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_train],
                'Class_id':[i[1] for i in extracted_image_id_train],
                'class_description':[i[2] for i in extracted_image_id_train]
                })
    export_csv_train = export_df_train.to_csv ('./extracted_info/image_id+class_id_train.csv', index = None, header=True)
    print ('(training data) Finished exporting to image_id+class_id_train.csv')

    #validation
    image_id_data_validation = pd.read_csv("./ImageSrc/validation/annotations-human.csv")
    validation_img_cnt = 0

    for i in range(len(image_id_data_validation.ImageID)):
        if (image_id_data_validation.LabelName[i] in extracted_class_id) and image_id_data_validation.Confidence[i]:
            extracted_image_id_validation.append([
                image_id_data_validation.ImageID[i],
                sel_class[extracted_class_id.index(image_id_data_validation.LabelName[i])][0],
                sel_class[extracted_class_id.index(image_id_data_validation.LabelName[i])][1]
                ])
            #print (food_list['food_name'][i])
            validation_img_cnt += 1

    print (str(validation_img_cnt)+' validation images extracted.')

    export_df_validation = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_validation],
                'Class_id':[i[1] for i in extracted_image_id_validation],
                'class_description':[i[2] for i in extracted_image_id_validation]
                })
    export_csv_validation = export_df_validation.to_csv ('./extracted_info/image_id+class_id_validation.csv', index = None, header=True)
    print ('(validation data) Finished exporting to image_id+class_id_validation.csv')

    #test
    image_id_data_test = pd.read_csv("./ImageSrc/test/annotations-human.csv")
    test_img_cnt = 0

    for i in range(len(image_id_data_test.ImageID)):
        if (image_id_data_test.LabelName[i] in extracted_class_id) and image_id_data_test.Confidence[i]:
            extracted_image_id_test.append([
                image_id_data_test.ImageID[i],
                sel_class[extracted_class_id.index(image_id_data_test.LabelName[i])][0],
                sel_class[extracted_class_id.index(image_id_data_test.LabelName[i])][1]
                ])
            #print (food_list['food_name'][i])
            test_img_cnt += 1

    print (str(test_img_cnt)+' test images extracted.')

    export_df_test = pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_id_test],
                'Class_id':[i[1] for i in extracted_image_id_test],
                'class_description':[i[2] for i in extracted_image_id_test]
                })
    export_csv_test = export_df_test.to_csv ('./extracted_info/image_id+class_id_test.csv', index = None, header=True)
    print ('(test data) Finished exporting to image_id+class_id_test.csv')


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
                class_id_data.class_description[idx_in_data]
                ])
            img_cnt += 1

    return [img_cnt,extracted_image_url]

def helper_export_df(extracted_image_url,train_val_test):
    export_df= pd.DataFrame({
                'Image_id':[i[0] for i in extracted_image_url],
                'Image_title':[i[1] for i in extracted_image_url],
                'Image_url':[i[2] for i in extracted_image_url],
                'Class_id':[i[3] for i in extracted_image_url],
                'class_description':[i[4] for i in extracted_image_url]
                })
    export_csv_train = export_df.to_csv ('./extracted_info/image_url_extracted_'+train_val_test+'.csv', index = None, header=True)

def image_url():

    extracted_image_url_train = []

    url_col_names = ['ImageID','Subset','OriginalURL','OriginalLandingURL','License','AuthorProfileURL','Author','Title','OriginalSize','OriginalMD5','Thumbnail300KURL']
    total_size = 9011220 - 1

############################################################
    #train
    if not os.path.isfile('./extracted_info/image_url_extracted_train.csv'):
        #read url file
        image_url_data = pd.read_csv('./ImageSrc/train/images.csv',usecols=['ImageID','Title','Thumbnail300KURL'])

        #read class id file
        class_id_data = pd.read_csv('./extracted_info/image_id+class_id_train.csv')
        class_id_data_format = ['Class_id','Image_id','class_description']

        [train_img_cnt,extracted_image_url_train] = helper_search(image_url_data,class_id_data)

        print (str(train_img_cnt)+' train images url extracted.')

        helper_export_df(extracted_image_url_train,'train')
        print ('(train data) Finished exporting to image_url_extracted_train.csv')

    else:
        print ('(train data) Already exported to image_url_extracted_train.csv')



########################################################################################
    #validaiton
    if not os.path.isfile('./extracted_info/image_url_extracted_validation.csv'):
        #read url file
        image_url_data = pd.read_csv('./ImageSrc/validation/images.csv',usecols=['ImageID','Title','Thumbnail300KURL'])

        #read class id file
        class_id_data = pd.read_csv('./extracted_info/image_id+class_id_validation.csv')
        class_id_data_format = ['Class_id','Image_id','class_description']

        [validation_img_cnt,extracted_image_url_validation] = helper_search(image_url_data,class_id_data)

        print (str(validation_img_cnt)+' validation images url extracted.')

        helper_export_df(extracted_image_url_validation,'validation')

        print ('(validation data) Finished exporting to image_url_extracted_validation.csv')

    else:
        print ('(validation data) Already exported to image_url_extracted_validation.csv')



########################################################################################
    #test
    if not os.path.isfile('./extracted_info/image_url_extracted_test.csv'):
        #read url file
        image_url_data = pd.read_csv('./ImageSrc/test/images.csv',usecols=['ImageID','Title','Thumbnail300KURL'])

        #read class id file
        class_id_data = pd.read_csv('./extracted_info/image_id+class_id_test.csv')
        class_id_data_format = ['Class_id','Image_id','class_description']

        [test_img_cnt,extracted_image_url_test] = helper_search(image_url_data,class_id_data)

        print (str(test_img_cnt)+' test images url extracted.')

        helper_export_df(extracted_image_url_test,'test')

        print ('(test data) Finished exporting to image_url_extracted_test.csv')

    else:
        print ('(test data) Already exported to image_url_extracted_test.csv')


def sort_csv_with_class():
    image_url_data = pd.read_csv('./extracted_info/image_url_extracted_train.csv')
    export_sorted_csv_train = image_url_data.sort_values(by=['Class_id']).to_csv ('./extracted_info/image_url_extracted_train.csv', index = None, header=True)

    image_url_data = pd.read_csv('./extracted_info/image_url_extracted_validation.csv')
    export_sorted_csv_validation = image_url_data.sort_values(by=['Class_id']).to_csv ('./extracted_info/image_url_extracted_validation.csv', index = None, header=True)

    image_url_data = pd.read_csv('./extracted_info/image_url_extracted_test.csv')
    export_sorted_csv_test = image_url_data.sort_values(by=['Class_id']).to_csv ('./extracted_info/image_url_extracted_test.csv', index = None, header=True)

def download(train_validation_test):
    image_url_data = pd.read_csv('./extracted_info/image_url_extracted_'+train_validation_test+'.csv')
    for i in range(len(image_url_data.Image_url)):
        name = "./data/"+train_validation_test+"/"+image_url_data.class_description[i]+"/"+str(i).zfill(6)
        url = image_url_data.Image_url[i]
        print ('Image '+str(i).zfill(6))

        try:
            print ('Downloading from '+url)
            print ('Saving to '+name)
            urllib.request.urlretrieve(url,name)
            print ('Success ...\n\n')
        except:
            print ('Failed ...\n\n')
    print ('Finished downloading '+train_validation_test+' images ...')

def resize_all():
    grandmother = os.listdir('./data') # train, test, val
    for i in grandmother:
        parent = os.listdir('./data/'+i) # bacon, ...
        for j in parent:
            child = os.listdir('./data/'+i+'/'+j) # images
            idx = 0
            for k in child:

                old_path = './data/'+i+'/'+j+'/'+k
                print ('old file path: '+old_path)
                new_path = './resized_data/'+i+'/'+j+'/'+str(idx).zfill(4)+'.jpg'
                idx += 1
                resize_im(old_path,new_path)
    print ('Finished resizing all images ...')

##############################################################
#main

#check if the first and second step is finished
if os.path.isfile('./extracted_info/image_id+class_id_test.csv') and os.path.isfile('./extracted_info/image_id+class_id_train.csv') and os.path.isfile('./extracted_info/image_id+class_id_validation.csv'):
    print ('First and second step finished ... ')
    [food_cnt,extracted_class_id,sel_class] = get_class_id()
    print ('Extracting URL ...')
    image_url()
    print ('FINISHED EXTRACTING URL  ... ')
    sort_csv_with_class()
    print ('SORTED')
else:
    print ('Running first and second step ...')
    image_id_csv()
    print ('First and second step finished ... ')
    print ('Extracting URL ...')
    image_url()
    print ('FINISHED EXTRACTING URL  ... ')
    sort_csv_with_class()
    print ('SORTED')

if not os.path.exists('./resized_data/'):
    os.mkdir('./resized_data/')
    print ('Made folder ./resized_data/')
    os.mkdir('./resized_data/train/')

    os.mkdir('./resized_data/validation/')

    os.mkdir('./resized_data/test/')

    for i in sel_class:
        os.mkdir('./resized_data/train/'+i[1])
    print ('Made folder ./resized_data/train/')
    for i in sel_class:
        os.mkdir('./resized_data/validation/'+i[1])
    print ('Made folder ./resized_data/validation/')
    for i in sel_class:
        os.mkdir('./resized_data/test/'+i[1])
    print ('Made folder ./resized_data/test/')
print ('Finished making folders')
resize_all()
#download('validation')
#download('test')
#download('train')
#third import images.csv
#go through the file
# if image_id in the list:
#	save the [image url, image description..., class id, class description]
