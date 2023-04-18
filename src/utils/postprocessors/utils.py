from .base_postprocessor import BasePostprocessor
from .knn_postprocessor import KNNPostprocessor
from .lof_postprocessor import LocalOutlierFactorPostprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .react_postprocessor import ReactPostprocessor
from .mds_postprocessor import MDSPostprocessor


def get_postprocessor(method, method_args):
    postprocessors = {
        'msp': BasePostprocessor,
        'knn': KNNPostprocessor,
        'lof': LocalOutlierFactorPostprocessor,
        'mls': MaxLogitPostprocessor,
        'odin': ODINPostprocessor,
        'react': ReactPostprocessor,
        'mds': MDSPostprocessor,
    }

    return postprocessors[method](method_args)
