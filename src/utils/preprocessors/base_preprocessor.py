import torchvision.transforms as tvs_trans

from .transform import Convert, interpolation_modes, normalization_dict


class BasePreprocessor():
    """For train dataset standard transformation."""
    def __init__(self, preprocessor_args):
        self.image_size = preprocessor_args['image_size']
        self.interpolation = interpolation_modes[preprocessor_args['interpolation']]
        normalization_type = preprocessor_args['normalization_type']
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.image_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(self.image_size, padding=4),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)
