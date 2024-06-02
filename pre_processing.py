import cv2 as cv
import os
import tifffile as tf
import histomicstk as htk
from skimage import transform

from crop_images import crop_images

def pre_processing(dir):
    files = os.listdir(dir)
    ResultPath1 = ''
    infer_path = ''
    infer = cv.imread(infer_path)
    infer = infer[:4096, :4096]
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(infer)
    image_data_mean, image_data_stddev = cv.meanStdDev(infer)
    print(image_data_mean)
    print(image_data_stddev)
    for file in files:  
        a, b = os.path.splitext(file) 
        this_dir = os.path.join(dir + file)
        img = cv.imread(this_dir, cv.IMREAD_GRAYSCALE)
        img = img[:4096, :4096]
        medianBlurImage = cv.medianBlur(img, 3)
        color_image = cv.cvtColor(medianBlurImage, cv.COLOR_GRAY2BGR)
        imNmzd = htk.preprocessing.color_normalization.reinhard(color_image, meanRef, stdRef)
        img_gray = cv.cvtColor(imNmzd, cv.COLOR_BGR2GRAY)
        image_data_mean, image_data_stddev = cv.meanStdDev(img_gray)
        image_data_mean_source, image_data_stddev_source = cv.meanStdDev(img)
        cv.imwrite(ResultPath1 + a + ".png", img_gray)

if __name__ == '__main__':
    _path = '' 
    pre_processing(_path)
