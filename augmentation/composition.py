from __future__ import print_function, division
from typing import Optional
import numpy as np
from skimage.filters import gaussian

class Compose(object):

    smooth_sigma = 2.0
    smooth_threshold = 0.5

    def __init__(self,
                 transforms: list = [],
                 input_size: tuple = (8,256,256),
                 smooth: bool = True,
                 keep_uncropped: bool = False,
                 keep_non_smoothed: bool = False,
                 additional_targets: Optional[dict] = None):

        self.transforms = transforms
        self.set_flip()

        self.input_size = np.array(input_size)
        self.sample_size = self.input_size.copy()
        self.set_sample_params()

        self.smooth = smooth
        self.keep_uncropped = keep_uncropped
        self.keep_non_smoothed = keep_non_smoothed

        if additional_targets is not None:
            self.additional_targets = additional_targets
        else: # initialize as an empty dictionary
            self.additional_targets = {}

    def set_flip(self):
        self.flip_aug = None
        flip_idx = None

        for i, t in enumerate(self.transforms):
            if t.__class__.__name__ == 'Flip':
                self.flip_aug = t
                flip_idx = i

        if flip_idx is not None:
            del self.transforms[flip_idx]

    def set_sample_params(self):
        for _, t in enumerate(self.transforms):
            self.sample_size = np.ceil(self.sample_size * t.sample_params['ratio']).astype(int)
            self.sample_size = self.sample_size + (2 * np.array(t.sample_params['add']))
        print('Sample size required for the augmentor:', self.sample_size)

    def smooth_edge(self, masks):
        smoothed_masks = masks.copy()

        for z in range(smoothed_masks.shape[0]):
            temp = smoothed_masks[z].copy()
            for idx in np.unique(temp):
                if idx != 0:
                    binary = (temp==idx).astype(np.uint8)
                    for _ in range(2):
                        binary = gaussian(binary, sigma=self.smooth_sigma, preserve_range=True)
                        binary = (binary > self.smooth_threshold).astype(np.uint8)

                    temp[np.where(temp==idx)]=0
                    temp[np.where(binary==1)]=idx

            smoothed_masks[z] = temp

        return smoothed_masks

    def center_crop(self, images, z_low=0):
        assert images.ndim in [3, 4]
        z_len, y_len, x_len = images.shape[-3:]
        margin_z = int((z_len - self.input_size[0]) // 2)
        margin_y = int((y_len - self.input_size[1]) // 2)
        margin_x = int((x_len - self.input_size[2]) // 2)

        z_low, z_high = margin_z, margin_z + self.input_size[0]
        y_low, y_high = margin_y, margin_y + self.input_size[1]
        x_low, x_high = margin_x, margin_x + self.input_size[2]

        if images.ndim == 3:
            return images[z_low:z_high, y_low:y_high, x_low:x_high]
        else:
            return images[:, z_low:z_high, y_low:y_high, x_low:x_high]

    def __call__(self, sample, random_state=np.random.RandomState()):
        # According to this blog post (https://www.sicara.ai/blog/2019-01-28-how-computer-generate-random-numbers):
        # we need to be careful when using numpy.random in multiprocess application as it can always generate the
        # same output for different processes. Therefore we use np.random.RandomState().
        sample['image'] = sample['image'].astype(np.float32)
        for name in self.additional_targets.keys():
            if self.additional_targets[name] == 'img':
                sample[name] = sample[name].astype(np.float32)

        ran = random_state.rand(len(self.transforms))
        for tid, t in enumerate(reversed(self.transforms)):
            if ran[tid] < t.p:
                sample = t(sample, random_state)

        # crop the data to the specified input size
        existing_keys = ['image'] + list(self.additional_targets.keys())
        for key in existing_keys:
            if self.keep_uncropped:
                new_key = 'uncropped_' + str(key)
                sample[new_key] = sample[key].copy()
            sample[key] = self.center_crop(sample[key])

        # flip augmentation
        if self.flip_aug is not None and random_state.rand() < self.flip_aug.p:
            sample = self.flip_aug(sample, random_state)

        # smooth mask contour
        if self.smooth:
            for key in self.additional_targets.keys():
                if self.additional_targets[key] == 'mask':
                    if self.keep_non_smoothed:
                        new_key = 'not_smoothed_' + str(key)
                        sample[new_key] = sample[key].copy()
                    sample[key] = self.smooth_edge(sample[key].copy())

        return sample
