# from PIL import Image, ImageOps
#
# desired_size = 640
# im_pth = "003743"
#
# im = Image.open(im_pth)
# old_size = im.size  # old_size[0] is in (width, height) format
#
# ratio = float(desired_size)/max(old_size)
# new_size = tuple([int(x*ratio) for x in old_size])
# # use thumbnail() or resize() method to resize the input image
#
# # thumbnail is a in-place operation
#
# # im.thumbnail(new_size, Image.ANTIALIAS)
#
# im = im.resize(new_size, Image.ANTIALIAS)
# # create a new image and paste the resized on it
#
# new_im = Image.new("RGB", (desired_size, desired_size))
# new_im.paste(im, ((desired_size-new_size[0])//2,
#                     (desired_size-new_size[1])//2))
#
# new_im.show()


import cv2

def resize_im(old_path,new_path):
    desired_size = 640
    im_pth = old_path

    im = cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    cv2.imwrite(new_path,new_im)
#cv2.imshow("image", new_im)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
