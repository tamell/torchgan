__all__ = ['EvaluationDistance']

# Work in progress

class EvaluationDistance:

    def __init__(self):
        self.arg_map = {}

    def set_arg_map(self, value):
        self.arg_map.update(value)

    def preprocess(self, x):
        raise NotImplementedError

    def calculate_distance(self, x1, x2):
        raise NotImplementedError

    def metric_ops(self, generator,  **kwargs):
        raise NotImplementedError

    def __call__(self, x1, x2):
        return self.calculate_distance(self.preprocess(x1), self.preprocess(x2))