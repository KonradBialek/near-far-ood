from .base_postprocessor import BasePostprocessor


def get_postprocessor(method, method_args):
    postprocessors = {
        'msp': BasePostprocessor,
    }

    return postprocessors[method](method_args)
