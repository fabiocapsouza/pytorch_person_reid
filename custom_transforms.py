import numpy as np
import cv2
import torch


class RandomTranslation(object):
    
    def __init__(self, translation_range=0.15, borderMode='BORDER_REPLICATE'):
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        elif isinstance(translation_range, (list, tuple)):
            if not len(translation_range) == 2: 
                raise ValueError("ranges must have length 2.")
        if not 0 <= translation_range[0] <= 1 or not 0 <= translation_range[1] <= 1:
            raise ValueError("ranges must be between 0 and 1")
        
        self.translation_range = translation_range
        
        if not hasattr(cv2, borderMode):
            raise ValueError("Border Mode {} not found in OpenCV.".format(borderMode))
        self.borderMode = getattr(cv2, borderMode)
        
    
    def __call__(self, inp):
        # arr must be a numpy array (H, W, C)
        # or a torch Tensor (C, H, W)
        if torch.is_tensor(inp):
            arr = inp.numpy().transpose(1,2,0)
        else:
            # if inp.dtype != np.float32:
            #     arr = inp.astype(np.float32)
            # else:
            #     arr = inp
            arr = inp
        H, W, _ = arr.shape
    
        h_range, w_range = self.translation_range
        # height shift
        dh = int(np.round(H * np.random.uniform(-h_range, h_range)))
        # width shift
        dw = int(np.round(W * np.random.uniform(-w_range, w_range)))

        M = np.array([[1, 0, dw],
                      [0, 1, dh]] , dtype=np.float32)
        arr = cv2.warpAffine(arr, M, (W, H), borderMode=self.borderMode)
        
        if torch.is_tensor(inp):
            return torch.from_numpy(arr).float()
        return arr
          


class PILToNdarray(object):
    def __call__(self, image):
        return np.fromstring(image.tobytes(), dtype=np.uint8).reshape((image.size[1], image.size[0], 3))