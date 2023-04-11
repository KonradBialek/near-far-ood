from .base_preprocessor import BasePreprocessor
from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(preprocessor_args, split):
    train_preprocessors = {
        'base': BasePreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[preprocessor_args['name']](preprocessor_args)
    else:
        return test_preprocessors[preprocessor_args['name']](preprocessor_args)
