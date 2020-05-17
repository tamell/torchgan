import torch
import torchvision
from torchgan.metrics.distance import EvaluationDistance

__all__ = ['FrechetDistance']

# Work in progress
class FrechetDistance(EvaluationDistance):
    """Compute the GAN's Frechet Distance.

    Given true data and a generator, calculate the Frechet distance between
    the generated dataset and the true one. Does not assume that GANs are
    image generating, but does assume that the encoding dimension is 1-dimensional.

    Inputs:
    sample_size: int, optional
          Number of generated samples to draw from the generator.
    classifier: torch Module, optional
          Encoder for the GAN.
    transform: func, optional
          A function to be applied to the output of the generator before comparison.
          """
    def __init__(self, sample_size=1, classifier=None, transform=None):
        super(FrechetDistance, self).__init__()
        if classifier is None:
            classifier = torchvision.models.inception_v3(True)
        self.classifier = classifier
        self.sample_size = sample_size
        self.transform = transform

    def preprocess(self, x):
        """Preprocess generator output before comparing the distance."""
        return self.transform(x) if self.transform is not None else x

    def calculate_distance(self, x1, x2):
        """Calculate the Frechet distance between two sets of samples."""
        feature_x1 = self.classifier(x1)
        feature_x2 = self.classifier(x2)

        mean1, cov1 = torch.mean(feature_x1, dim=1), self.__cov(feature_x1)
        mean2, cov2 = torch.mean(feature_x2, dim=1), self.__cov(feature_x2)

        mean_comp = torch.norm(mean2 - mean1, 2)
        cov_comp = torch.trace(cov1 + cov2 - 2*torch.sqrt(cov1 * cov2))
        return mean_comp + cov_comp

    def metric_ops(self, generator, device, true_data):
        """Generate samples using GAN and calculate distance to true samples."""
        noise = torch.randn(self.sample_size, generator.encoding_dims, device=device)
        gen_data = generator(noise)
        score = self.__call__(gen_data, true_data)

        return score

    def __cov(self, x):
        """Calculate the covariance matrix of samples.

        Assumes the shape (n_dimensions, n_samples).
        """
        x_mean = torch.mean(x, dim=1)
        n_samples = x.size(1)
        return torch.true_divide(x_mean.mul(torch.transpose(x_mean, 0, 1)), n_samples - 1)
