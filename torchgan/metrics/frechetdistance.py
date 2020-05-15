import torch
import torchvision
from torchgan.metrics.distance import EvaluationDistance

__all__ = ['FrechetDistance']


class FrechetDistance(EvaluationDistance):

    def __init__(self, classifier, sample_size=1):
        super(FrechetDistance, self).__init__()
        if classifier is None:
            classifier = torchvision.models.inception_v3(True)
        self.classifier = classifier
        self.sample_size = sample_size

    def set_arg_map(self, value):
        pass

    def preprocess(self, x):
        return x

    def calculate_distance(self, x1, x2):
        feature_x1 = self.classifier(x1)
        feature_x2 = self.classifier(x2)

        mean1, cov1 = torch.mean(feature_x1, dim=1), self.__cov(feature_x1, dim=1)
        mean2, cov2 = torch.mean(feature_x2, dim=1), self.__cov(feature_x2, dim=1)

        mean_comp = torch.norm(mean2 - mean1, 2)
        cov_comp = torch.trace(cov1 + cov2 - 2*torch.sqrt(cov1 * cov2))
        return mean_comp + cov_comp

    def metric_ops(self, generator, discriminator, **kwargs):
        pass

    def __cov(self, X, dim=None):
        return torch.Tensor(0)
