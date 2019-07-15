# APS360 PROJECT

Food Recognition

**Folder Explanation**

new_resources: collected in the third week (csv files) (currently using this dataset)

resized_images: contains all images with box information (640*640) (currently using this dataset)

box info: contains (resized box info) csv files of [image id, image url, box information...]

images: contains raw images

old_extracted_info: collected in the first week (abandoned because there is no box)

google_crawler: collected in the second week (abandoned because there is no box)

old_image_collected: downloaded images with extracted_info (abandoned)

**Image Collection**

plan: extract the food part of google open Images

Google Open Images: https://github.com/widemeadows/openimages-dataset

**Step 1**, download class and description from https://storage.googleapis.com/openimages/2017_07/classes_2017_07.tar.gz
since this file contains all possible classes in the google open image dataset, we need to look for image classes with the specific food we want

extracted: [class_id, class_description]

**Step 2**, download annotations-human labels from
https://storage.googleapis.com/openimages/2017_07/annotations_human_2017_07.tar.gz
since this file only has image_id with its labelled class_id
what we did is to record the image_id with class_id found from the first step

extracted: [image_id, class_id, class_description]

**Step 3**, download images url information from
https://storage.googleapis.com/openimages/2017_07/images_2017_07.tar.gz
this file contains information such as, image id, image url, image title ...
we went through the file and extracted image url with corresponding image_id from the second step

extracted: [image_id, image_title, image_url]

**Step 4**: download all images from the extracted url into folders with corresponding labels (use urllib command)

**Step 5**: resize all images to 640 by 640 pixels using black padding in order to keep the original ratio (use openCV)

**Next Step**: check all images
