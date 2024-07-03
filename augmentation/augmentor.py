from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from typing import Optional
import numpy as np

class DataAugment(object, metaclass=ABCMeta):
    def __init__(self,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):
        super().__init__()

        assert p >= 0.0 and p <=1.0
        self.p = p
        self.sample_params = {
            'ratio': np.array([1.0, 1.0, 1.0]),
            'add': np.array([0, 0, 0])}

        if additional_targets is not None:
            self.additional_targets = additional_targets
        else: # initialize as an empty dictionary
            self.additional_targets = {}

        self.skip_targets = skip_targets

    @abstractmethod
    def set_params(self):

        raise NotImplementedError

    @abstractmethod
    def __call__(self, sample, random_state=None):

        raise NotImplementedError
