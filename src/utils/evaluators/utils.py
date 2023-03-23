from .base_evaluator import BaseEvaluator
from .ood_evaluator import OODEvaluator


def get_evaluator(eval, eval_args):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
    }
    return evaluators[eval](eval_args)
