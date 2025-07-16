import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    # def loss(self, y_true, y_pred):
    #     ndims = len(list(y_pred.size())) - 2
    #     vol_axes = list(range(2, ndims + 2))
    #     top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    #     bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    #     dice = torch.mean(top / bottom)
    #     return -dice

    def loss(self, array1, array2, labels=torch.arange(1,36)):

        """
        Computes the dice overlap between two arrays for a given set of integer labels.

        Parameters:
            array1: Input array 1.
            array2: Input array 2.
            labels: List of labels to compute dice on. If None, all labels will be used.
            include_zero: Include label 0 in label list. Default is False.
        """
        if labels is None:
            labels = np.concatenate([np.unique(a) for a in [array1, array2]])
            labels = np.sort(np.unique(labels))
            labels = np.delete(labels, np.argwhere(labels == 0))

        dicem = torch.zeros(len(labels),dtype=torch.float,device=torch.device('cuda'))
        for idx, label in enumerate(labels):
            top = 2 * torch.sum(torch.logical_and(array1 == label, array2 == label))
            bottom = torch.sum(array1 == label) + torch.sum(array2 == label)
            bottom = torch.max(bottom.type_as(dicem), torch.cuda.FloatTensor([1E-10]))  # add epsilon to prevent division by zero
            # bottom = torch.max(bottom, torch.finfo(torch.float32).eps)  # add epsilon
            dicem[idx] = top / bottom
        return 1-dicem.mean()
    def loss_single_float(self, array1, array2, labels=1):

        """
        Computes the dice overlap between two arrays for a given set of float labels.
        """

        # dicem = torch.zeros(len(labels),dtype=torch.float,device=torch.device('cuda'))
        top = 2 * torch.sum(array1*array2)
        bottom = torch.sum(array1+array2) 
        # bottom = torch.max(bottom.type_as(dicem), torch.cuda.FloatTensor([1E-10]))  # add epsilon to prevent division by zero, by why would there be zero?
        dice = top / bottom

        # if -dice > -0.01:
        #     print('spotted zero')
        #     print(top)
        #     print(bottom)
        # else:
        #     print(dice)
        return 1-dice

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
