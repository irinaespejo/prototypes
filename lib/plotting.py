import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional
import torch
from torch import Tensor
import numpy as np


def make_plot(true_x:Tensor, true_y:Tensor, train_x:Tensor, train_y:Tensor, t:float,f,mean: Optional[Tensor]=None, std: Optional[Tensor]=None, new_x: Optional[Tensor]=None, acq:Optional[Tensor] = None):
    x_max = torch.max(true_x)
    x_min = torch.min(true_x)
        
    if acq is None: #plot only fit no acquisition
        if mean is not None:
            plt.plot(true_x.detach().numpy(), mean, color='blue', label='mean');
        if std is not None:
            plt.fill_between(true_x.detach().numpy(), mean-std, mean+std, color='lightblue', label='std');
        
        fig = plt.figure(figsize=(12, 7))
        plt.plot(true_x, true_y, color='black', label='true', linewidth=1.5);
        plt.scatter(train_x.detach().numpy(), train_y.detach().numpy(), color='black', marker='o', label='train',  s=10)
        plt.vlines(train_x.detach().numpy(),np.max(true_y.detach().numpy()),np.min(true_y.detach().numpy()), color='grey', linewidth=0.5)

        if new_x is not None:
            plt.scatter(new_x.detach().numpy(), f(new_x).detach().numpy(), color='red', marker='o', s=10)
            plt.vlines(new_x.detach().numpy(),np.max(true_y.detach().numpy()),np.min(true_y.detach().numpy()), color='red', linewidth=0.5)
        
        
        plt.hlines(t, x_max, x_min, color='grey',linestyles='dashdot', label=f'threshold={t}', linewidth=2);
        #plt.ylim([-1.1,1.1])
        #mask = (torch.abs(true_y-t) < 0.5*10E-2).nonzero()
        #plt.scatter(true_x[mask].detach().numpy(), f(true_x[mask]).detach().numpy() , color='black', marker='o', label='true LS', facecolors='none')
        lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.xlabel('x')
        plt.ylabel('y')
        
    else: #plot acquisition fucntion
        fig = plt.figure(figsize=(12, 7))
        axes = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(axes[0])
        ax1 = plt.subplot(axes[1])

        #plot true and threshols
        ax0.plot(true_x, true_y, color='black', linestyle='--', label='true');
        ax0.vlines(train_x.detach().numpy(),np.max(true_y.detach().numpy()),np.min(true_y.detach().numpy()), color='grey', linewidth=0.5)
        ax0.hlines(t, x_min, x_max, color='grey', label=f'threshold={t}', linewidth=2);
        #ax0.set_ylim([-1.1,1.1])
        mask = (torch.abs(true_y-t) < 0.5*10E-2).nonzero()
        lgd = ax0.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        
        if mean is not None:
            ax0.plot(true_x.detach().numpy(), mean, color='blue', label='mean');
        if std is not None:
            ax0.fill_between(true_x.flatten().detach().numpy(), mean-std, mean+std, color='lightblue', label='std');
        
        
        ax0.scatter(train_x.detach().numpy(), train_y.detach().numpy(), color='black', marker='o', label='train', s=10)
        ax0.scatter(true_x[mask].detach().numpy(), f(true_x[mask]).detach().numpy() , color='black', marker='o', label='true LS', facecolors='none', s=10)

        if new_x is not None:
            ax0.scatter(new_x.detach().numpy(), f(new_x).detach().numpy(), color='red', marker='o', s=10)
            ax0.vlines(new_x.detach().numpy(),np.max(true_y.detach().numpy()),np.min(true_y.detach().numpy()), color='red', linewidth=0.5)
        
        # acq plot
        acq = acq.detach().numpy()
        mask = np.isfinite(acq)
        acq = acq[mask]
        # plot
        ax1.plot(true_x, acq, color="orange")
        ax1.set_xlabel("x")
        ax1.set_ylabel("acq(x)")
        ax1.set_yscale("log")
        vertical = ax1.axvline(new_x.detach().numpy(), c="red")
        lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.subplots_adjust(hspace=0.0)


def make_plot_2d(mesh:Tensor, true_y:Tensor, train_x:Tensor, train_y:Tensor, t:float,f,mean: Optional[Tensor]=None, std: Optional[Tensor]=None, new_x: Optional[Tensor]=None, ):
    """
    Plot GP posterior fit to data with the option of plotting side by side acquisition function
    """

    # true function + thresholds
    x1, x2 = mesh
    line0 = plt.contour(
            x1,
            x2,
            true_y.reshape(10,10),
            t,
            colors="white",
            linestyles="dotted",
            label="true contour",
        )

