import random

import numpy as np

from rastervision.augmentor import Augmentor
from rastervision.core import (TrainingData, Box)


class OversamplingAugmentor(Augmentor):
    """Increase the number of samples of a target class and reduce 
    the number of the non-target class to achieve a balanced training ratio.
    Scikit-learn is in RV requirements so can be used.
    """

    def __init__(self, aug_prob):
        self.aug_prob = aug_prob

    def process(self, training_data, tmp_dir):
        augmented = TrainingData()
        nodata_aug_prob = self.aug_prob

        for chip, window, labels in training_data:
            
            # If negative chip consider dropping it
            if len(labels) == 0 and random.uniform(0, 1) < nodata_aug_prob:
                chip = np.copy(chip)
                # drop chip

            # If positive chip duplicate and augment it

            # Ensure these are balanced...
                
            # all chips get returned in the augmented object
            augmented.append(chip, window, labels)

        return augmented
