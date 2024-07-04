import cv2 as cv
import os
import tifffile as tf
import histomicstk as htk
from histomicstk.preprocessing import color_conversion
from skimage import transform
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma

from crop_images import crop_images

def pre_processing(dir):
    files = os.listdir(dir)
    ResultPath1 = ''
    infer_path = ''
    infer = cv.imread(infer_path)
    height, width = infer.shape[:2]
    new_width = width // 8
    new_height = height // 8
    resized_img = cv.resize(infer, (new_width, new_height))
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(resized_img)
    image_data_mean, image_data_stddev = cv.meanStdDev(resized_img)
    print(image_data_mean)
    print(image_data_stddev)
    for file in files:
        a, b = os.path.splitext(file)
        this_dir = os.path.join(dir + file)
        img = tf.imread(this_dir, cv.IMREAD_GRAYSCALE)
        if mask_out is not None:
            mask_out = mask_out[..., None]
            img = np.ma.masked_array(
                im_lab, mask=np.tile(mask_out, (1, 1, 3)))
        if (src_mu is None) or (src_sigma is None):
            src_mu = [im_lab[..., i].mean() for i in range(3)]
            src_sigma = [im_lab[..., i].std() for i in range(3)]
        for i in range(3):
            im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]
        for i in range(3):
            im_lab[:, :, i] = im_lab[:, :, i] * meanRef[i] + stdRef[i]
        im_normalized = color_conversion.lab_to_rgb(im_lab)
        im_normalized[im_normalized > 255] = 255
        im_normalized[im_normalized < 0] = 0
        if mask_out is not None:
            im_normalized = im_normalized.data
            for i in range(3):
                original = color_image[:, :, i].copy()
                new = im_normalized[:, :, i].copy()
                original[np.not_equal(mask_out[:, :, 0], True)] = 0
                new[mask_out[:, :, 0]] = 0
                im_normalized[:, :, i] = new + original
        im_normalized = im_normalized.astype(np.uint8)
        img_gray = cv.cvtColor(im_normalized, cv.COLOR_BGR2GRAY)
        image_data_mean, image_data_stddev = cv.meanStdDev(img_gray)
        image_data_mean_source, image_data_stddev_source = cv.meanStdDev(img)
        print(a)
        print(image_data_mean_source)
        print(image_data_stddev_source)
        print(image_data_mean)
        print(image_data_stddev)
        tf.imwrite(ResultPath1 + a + ".tif", img_gray)
        
if __name__ == '__main__':
    _path = ''
    pre_processing(_path)
