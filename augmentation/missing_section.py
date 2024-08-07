from __future__ import print_function, division
from typing import Optional

import math
import numpy as np
from .augmentor import DataAugment

class MissingSection(DataAugment):

    def __init__(self,
                 num_sections: int = 2,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(MissingSection, self).__init__(p, additional_targets, skip_targets)
        self.num_sections = num_sections
        self.set_params()

    def set_params(self):

        self.sample_params['add'] = [int(math.ceil(self.num_sections / 2.0)), 0, 0]

    def missing_section(self, images, idx):
        return np.delete(images, idx, 0)

    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        if images.shape[0] == 1:
            # no missing-section augmentation for 2D images
            return sample

        # select slices to discard
        idx = random_state.choice(np.array(range(1, images.shape[0]-1)),
                                  self.num_sections, replace=False)

        sample['image'] = self.missing_section(images, idx)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self.missing_section(sample[key].copy(), idx)
        return sample
