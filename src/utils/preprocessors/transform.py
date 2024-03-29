import torchvision.transforms as tvs_trans

normalization_dict = {
    'cifar10': ([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
    'cifar100': ([0.48042983, 0.44819681, 0.39755555], [0.2764398, 0.26888656, 0.28166855]),
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'mnist': ([0.13062754273414612, 0.13062754273414612, 0.13062754273414612], [0.30810779333114624, 0.30810779333114624, 0.30810779333114624]),
    'fashionmnist': ([0.28604060411453247, 0.28604060411453247, 0.28604060411453247], [0.3530242443084717, 0.3530242443084717, 0.3530242443084717]),
    'notmnist': ([0.4239663035214087, 0.4239663035214087, 0.4239663035214087], [0.4583350861943875, 0.4583350861943875, 0.4583350861943875]),
    'dtd': ([0.52875836, 0.4730212, 0.4247069], [0.26853561, 0.25950334, 0.26667375]),
    'svhn': ([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]),
    'tin': ([0.48023694, 0.44806704, 0.39750364], [0.27643643, 0.26886328, 0.28158993]),
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
    'bicubic': tvs_trans.InterpolationMode.BICUBIC,
}

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)
