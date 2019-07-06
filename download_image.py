#google image data info
# use the following link
#https://github.com/widemeadows/openimages-dataset


import pandas as pd

sel_class= [] # this list will store extracted extracted [class_id,class_description]
extracted_class_id = [] # this list is for stroing class_id only
extracted_image_id_train = []
extracted_image_id_validation = []
extracted_image_id_test = []

#first import class-descriptions.csv
#look for classes with food and add [class id, class description] to the list
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


#second import annotations-human.csv
#go through the file
# if class_id in the list:
#	save the [image id, class id, class description]

image_id_col_names = ['image_id','human','class_id','confidence_level']
# training
image_id_data_train = pd.read_csv("./ImageSrc/train/annotations-human.csv",names=image_id_col_names,dtype=object)
train_img_cnt = 0

for i in range(len(image_id_data_train.image_id)):
    if (image_id_data_train.class_id[i] in extracted_class_id):
        extracted_image_id_train.append([
            image_id_data_train.image_id[i],
            sel_class[extracted_class_id.index(image_id_data_train.class_id[i])][0],
            sel_class[extracted_class_id.index(image_id_data_train.class_id[i])][1],
            ])
        #print (food_list['food_name'][i])
        train_img_cnt += 1

print (str(train_img_cnt)+' training images extracted.')

export_df_train = pd.DataFrame({
            'Image id':[i[0] for i in extracted_image_id_train],
            'Class_id':[i[1] for i in extracted_image_id_train],
            'class_description':[i[2] for i in extracted_image_id_train]
            })
export_csv_train = export_df_train.to_csv ('image_id+class_id_train.csv', index = None, header=True)
print ('(training data) Finished exporting to image_id+class_id_train.csv')

#validation
image_id_data_validation = pd.read_csv("./ImageSrc/validation/annotations-human.csv",names=image_id_col_names,dtype=object)
validation_img_cnt = 0

for i in range(len(image_id_data_validation.image_id)):
    if (image_id_data_validation.class_id[i] in extracted_class_id):
        extracted_image_id_validation.append([
            image_id_data_validation.image_id[i],
            sel_class[extracted_class_id.index(image_id_data_validation.class_id[i])][0],
            sel_class[extracted_class_id.index(image_id_data_validation.class_id[i])][1],
            ])
        #print (food_list['food_name'][i])
        validation_img_cnt += 1

print (str(validation_img_cnt)+' validation images extracted.')

export_df_validation = pd.DataFrame({
            'Image id':[i[0] for i in extracted_image_id_validation],
            'Class_id':[i[1] for i in extracted_image_id_validation],
            'class_description':[i[2] for i in extracted_image_id_validation]
            })
export_csv_validation = export_df_validation.to_csv ('image_id+class_id_validation.csv', index = None, header=True)
print ('(validation data) Finished exporting to image_id+class_id_validation.csv')

#test
image_id_data_test = pd.read_csv("./ImageSrc/test/annotations-human.csv",names=image_id_col_names,dtype=object)
test_img_cnt = 0

for i in range(len(image_id_data_test.image_id)):
    if (image_id_data_test.class_id[i] in extracted_class_id):
        extracted_image_id_test.append([
            image_id_data_test.image_id[i],
            sel_class[extracted_class_id.index(image_id_data_test.class_id[i])][0],
            sel_class[extracted_class_id.index(image_id_data_test.class_id[i])][1],
            ])
        #print (food_list['food_name'][i])
        test_img_cnt += 1

print (str(test_img_cnt)+' test images extracted.')

export_df_test = pd.DataFrame({
            'Image id':[i[0] for i in extracted_image_id_test],
            'Class_id':[i[1] for i in extracted_image_id_test],
            'class_description':[i[2] for i in extracted_image_id_test]
            })
export_csv_test = export_df_test.to_csv ('image_id+class_id_test.csv', index = None, header=True)
print ('(test data) Finished exporting to image_id+class_id_test.csv')
#third import images.csv
#go through the file
# if image_id in the list:
#	save the [image url, image description..., class id, class description]
