from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch.distributions.normal import Normal
from typing import Optional
import torch
from torch import Tensor


class MaximumEntropySearch_LevelSetEstimation(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        threshold: Tensor,
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model, objective=objective)
        # self.maximize = maximize
        self.thr = threshold
        # self.thresholds = [-1E10] + self.thr.tolist() + [1E10]
        # self.pairs_thresholds = list(zip(self.thresholds, self.thresholds[1:]))
        # self.pairs_thresholds = [torch.Tensor(pair) for pair in self.pairs_thresholds]

    def forward(self, X: Tensor):
        # THIS IS FOR Q = 1,
        #print(f'##### X {X.shape}')
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2] # HEEEEEREEEE NEXT
        #print(f'##### batch_shape {batch_shape}')
       
        #print(f'##### mean before view {posterior.mean.shape}')
        mean = posterior.mean.view(batch_shape) #NEXT THIS NEEDS TO BE RESHAPED WHEN Q>1
        #print(f'##### mean {mean.shape}')
        std = torch.square(posterior.variance.view(batch_shape))
        #print(f'##### std {std.shape}')
        normal = Normal(mean, std)
        
        p_below = normal.cdf(self.thr)
        #print(f'##### p_below {p_below.shape}')
     
        distr = torch.distributions.Bernoulli(probs=(1-p_below))
        #print(f'##### distr {distr}')
        entropy = distr.entropy().view(batch_shape)
        #print(f'##### entropy {entropy.shape}')
        
        return entropy