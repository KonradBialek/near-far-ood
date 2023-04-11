import torchvision.transforms as tvs_trans

from .base_preprocessor import BasePreprocessor
from .transform import Convert


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, preprocessor_args):
        super(TestStandardPreProcessor, self).__init__(preprocessor_args)
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.image_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])
