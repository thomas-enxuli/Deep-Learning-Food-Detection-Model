# APS360 PROJECT

Food Recognition

**Image Collection**

plan: extract the food part of google open Images

Google Open Images: https://github.com/widemeadows/openimages-dataset

**First**, download class and description from https://storage.googleapis.com/openimages/2017_07/classes_2017_07.tar.gz
since this file contains all possible classes in the google open image dataset, we need to look for image classes with the specific food we want

extracted: [class_id, class_description]

**Second**, download annotations-human labels from
https://storage.googleapis.com/openimages/2017_07/annotations_human_2017_07.tar.gz
since this file only has image_id with its labelled class_id
what we did is to record the image_id with class_id found from the first step

extracted: [image_id, class_id, class_description]

**Third**, download images url information from
https://storage.googleapis.com/openimages/2017_07/images_2017_07.tar.gz
this file contains information such as, image id, image url, image title ...
we went through the file and extracted image url with corresponding image_id from the second step

extracted: [image_id, image_title, image_url]

**Next Step**: download all images from the extracted url into folders with corresponding labels (use wget command)
