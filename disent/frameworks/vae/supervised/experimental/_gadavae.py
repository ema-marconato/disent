#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class GuidedAdaVae(AdaVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(AdaVae.cfg):
        anchor_ave_mode: str = 'average'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # how the anchor is averaged
        assert cfg.anchor_ave_mode in {'thresh', 'average'}

    def hook_intercept_zs(self, zs_params: Sequence['Params']) -> Tuple[Sequence['Params'], Dict[str, Any]]:
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        a_z_params, p_z_params, n_z_params = zs_params

        # get distributions
        a_d_posterior, _ = self.params_to_dists(a_z_params)
        p_d_posterior, _ = self.params_to_dists(p_z_params)
        n_d_posterior, _ = self.params_to_dists(n_z_params)

        # get deltas
        a_p_deltas = AdaVae.compute_kl_deltas(a_d_posterior, p_d_posterior, symmetric_kl=self.cfg.symmetric_kl)
        a_n_deltas = AdaVae.compute_kl_deltas(a_d_posterior, n_d_posterior, symmetric_kl=self.cfg.symmetric_kl)

        # shared elements that need to be averaged, computed per pair in the batch.
        old_p_shared_mask = AdaVae.compute_shared_mask(a_p_deltas)
        old_n_shared_mask = AdaVae.compute_shared_mask(a_n_deltas)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = compute_constrained_masks(a_p_deltas, old_p_shared_mask, a_n_deltas, old_n_shared_mask)

        # make averaged variables
        pa_z_params, p_z_params = AdaVae.compute_averaged(a_z_params, p_z_params, p_shared_mask, self._compute_average_fn)
        na_z_params, n_z_params = AdaVae.compute_averaged(a_z_params, n_z_params, n_shared_mask, self._compute_average_fn)
        ave_params = self.latents_handler.encoding_to_params(self._compute_average_fn(pa_z_params.mean, pa_z_params.logvar, pa_z_params.mean, pa_z_params.logvar))

        anchor_ave_logs = {}
        if self.cfg.anchor_ave_mode == 'thresh':
            # compute anchor average using the adaptive threshold
            ave_shared_mask = p_shared_mask * n_shared_mask
            ave_params, _ = AdaVae.compute_averaged(a_z_params, ave_params, ave_shared_mask, self._compute_average_fn)
            anchor_ave_logs['ave_shared'] = ave_shared_mask.sum(dim=1).float().mean()

        new_args = ave_params, p_z_params, n_z_params
        return new_args, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
            **anchor_ave_logs,
        }
    

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def compute_constrained_masks(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask):
    # number of changed factors
    p_shared_num = torch.sum(p_shared_mask, dim=1, keepdim=True)
    n_shared_num = torch.sum(n_shared_mask, dim=1, keepdim=True)

    # POSITIVE SHARED MASK
    # order from smallest to largest
    p_sort_indices = torch.argsort(p_kl_deltas, dim=1)
    # p_shared should be at least n_shared
    new_p_shared_num = torch.max(p_shared_num, n_shared_num)

    # NEGATIVE SHARED MASK
    # order from smallest to largest
    n_sort_indices = torch.argsort(n_kl_deltas, dim=1)
    # n_shared should be at most p_shared
    new_n_shared_num = torch.min(p_shared_num, n_shared_num)

    # COMPUTE NEW MASKS
    new_p_shared_mask = torch.zeros_like(p_shared_mask)
    new_n_shared_mask = torch.zeros_like(n_shared_mask)
    for i, (new_shared_p, new_shared_n) in enumerate(zip(new_p_shared_num, new_n_shared_num)):
        new_p_shared_mask[i, p_sort_indices[i, :new_shared_p]] = True
        new_n_shared_mask[i, n_sort_indices[i, :new_shared_n]] = True

    # return masks
    return new_p_shared_mask, new_n_shared_mask


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

