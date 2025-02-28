"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.transform import resize
from fastmri.data.poisson_disc_variable_density import PoissonSampler


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        num_offsets=8,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset, num_offsets, seed
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class MaskFunc3D(MaskFunc):
    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
        num_offsets=16,
        seed = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)

        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask_3D(
                num_cols, acceleration, offset, num_low_frequencies, shape, seed # num_offsets, seed
            ),
            shape
        )
        center_mask = torch.zeros_like(acceleration_mask)
        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        return torch.from_numpy(mask.astype(np.float32))

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with exact acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class EquispacedMaskFractionFunc3D(MaskFunc3D):
    def __init__(self, center_fractions, accelerations, allow_any_combination=False, seed=None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.offset_mask = 0

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if self.offset_mask > 3:
            self.offset_mask = 0
        mask = np.zeros((shape[-3], shape[-2]))
        mask_offset = self.offset_mask % 2
        phase_offset = self.offset_mask // 2
        mask[mask_offset::3, phase_offset::3] = 1.0        
        self.offset_mask += 1
        return mask

class EquispacedMaskFractionFunc3D(MaskFunc3D):
    def __init__(self, center_fractions, accelerations, allow_any_combination=False, seed=None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.offset_mask = 0

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if self.offset_mask > 3:
            self.offset_mask = 0
        mask = np.zeros((shape[-3], shape[-2]))
        mask_offset = self.offset_mask % 2
        phase_offset = self.offset_mask // 2
        mask[mask_offset::3, phase_offset::3] = 1.0        
        self.offset_mask += 1
        return mask


class EquispacedMaskFractionFuncCenterDense3D(MaskFunc3D):
    def __init__(self, center_fractions, accelerations, allow_any_combination=False, seed=None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.offset_mask = 0
        
    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed
    ) -> np.ndarray:
        if self.offset_mask > 7:
            self.offset_mask = 0
        # print("WARNING: Offset is alsways zero for subsampling!")
        # self.offset_mask = 0
        mask_offset = [0, 2, 1, 3, 0, 2, 1, 3][self.offset_mask]
        phase_offset = [0, 0, 1, 1, 2, 2, 3, 3][self.offset_mask]
        mask = np.zeros((shape[-3], shape[-2]))
        mask[mask_offset::4, phase_offset::4] = 1.0
        center_offset = 5 + self.offset_mask % 2
        phase_offset = 32 + (self.offset_mask // 2) % 2
        mask[center_offset:center_offset+10, phase_offset:phase_offset+62] = 0
        mask[center_offset:center_offset+10:2, phase_offset:phase_offset+62:2] = 1
        self.offset_mask = self.offset_mask + 1
        return mask

class EquispacedMaskFractionFuncCenterFull3D(MaskFunc3D):
    def __init__(self, center_fractions, accelerations, allow_any_combination=False, seed=None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.offset_mask = 0
        
    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed
    ) -> np.ndarray:
        if self.offset_mask > 7:
            self.offset_mask = 0
        mask_offset = [0, 2, 1, 3, 0, 2, 1, 3][self.offset_mask]
        phase_offset = [0, 0, 1, 1, 2, 2, 3, 3][self.offset_mask]
        mask = np.zeros((shape[-3], shape[-2]))
        mask[mask_offset::4, phase_offset::4] = 1.0
        center_offset = 5 + self.offset_mask % 2
        phase_offset = 32 + (self.offset_mask // 2) % 2
        mask[center_offset:center_offset+10, phase_offset:phase_offset+62] = 1
        self.offset_mask = self.offset_mask + 1
        return mask


class VariableDensitiyMask3D(MaskFunc3D):
    def __init__(self, accelerations, allow_any_combination=False, seed=None):
        super().__init__([0], accelerations, allow_any_combination, seed)
        self.rng_new = np.random.default_rng(seed)
        self.num_samples = 300

    def draw_samples(self, shape, acceleration):
        tol = 0.1
        while True:
            s = self.rng_new.multivariate_normal([shape[1] // 2, shape[2] // 2], [[15, 0], [0, 500]], self.num_samples)
            s[:, 0] = np.clip(s[:, 0], 0, shape[1] - 1)
            s[:, 1] = np.clip(s[:, 1], 0, shape[2] - 1)
            s = s.astype(int)
            mask = np.zeros(shape[1:-1], dtype=bool)
            mask[s[:, 0], s[:, 1]] = True
            # Dense center sampling
            mask[int(mask.shape[0] * 3 / 8):int(mask.shape[0] * 5 / 8),
                 int(mask.shape[1] * 9 / 20):int(mask.shape[1] * 11 / 20)] = True
            R = 1 / (np.sum(mask) / (mask.shape[0] * mask.shape[1]))
            if R > acceleration:
                self.num_samples += 50
            elif R < acceleration:
                self.num_samples -= 50
            if np.abs(acceleration - R) < tol:
                return mask

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed,
    ) -> np.ndarray:
        self.rng_new = np.random.default_rng(seed)
        mask = self.draw_samples(shape, acceleration)
        return mask[None]


class PoissonDensitiyMask3D(MaskFunc3D):
    def __init__(self, accelerations, allow_any_combination=False, seed=None, cached=True):
        super().__init__([0], accelerations, allow_any_combination, seed)
        self.rng_new = np.random.default_rng(seed)
        self.poisson_radius = 0.0125
        self.rng_new = np.random.default_rng(seed)
        self.cached = cached
        if cached:
            self.poisson_masks = np.load(r"C:\Users\follels\Documents\fastMRI\cache\poisson_disc_masks\masks.npy")

    def poisson_disc_calculation(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed,
    ) -> np.ndarray:
        # Round to closest even number
        self.rng_new = np.random.default_rng(seed)
        inner_shape = ((shape[1] // 2) & ~1, (shape[2] // 2) & ~1)
        accu_inner = np.zeros((inner_shape[0], inner_shape[1]), dtype=np.int32)
        PS = PoissonSampler(inner_shape[0], inner_shape[1], 5, 2, 42, 0, 0.7)
        mask_inner = PS.generate(self.rng_new, accu_inner)

        outer_shape = (shape[1] & ~1, shape[2] & ~1)
        accu_outer = np.zeros((outer_shape[0], outer_shape[1]), dtype=np.int32)
        PS = PoissonSampler(outer_shape[0], outer_shape[1], 8, 4, 42, 0, 0.7)
        mask_combined = PS.generate(self.rng_new, accu_outer)
        mask_combined[mask_combined.shape[0] // 2 - inner_shape[0] // 2:mask_combined.shape[0] // 2 + inner_shape[0] // 2,
                      mask_combined.shape[1] // 2 - inner_shape[1] // 2:mask_combined.shape[1] // 2 + inner_shape[1] // 2] = mask_inner
        # For uneven shapes
        final_mask = np.zeros(shape[1:-1])
        final_mask[0:mask_combined.shape[0], 0:mask_combined.shape[1]] = mask_combined
        return final_mask

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        seed,
    ) -> np.ndarray:
        if self.cached:
            self.rng_new = np.random.default_rng(seed)
            case_idx = self.rng_new.integers(0, 100)
            mask = self.poisson_masks[case_idx]
        else:
            mask = self.poisson_disc_calculation(num_cols, acceleration, offset, num_low_frequencies, shape, seed)
        return mask
            

class CartesianOffsetMask3D(MaskFunc3D):
    def __init__(self, accelerations, allow_any_combination=False, seed=None):
        super().__init__([0], accelerations, allow_any_combination, seed)
        self.rng_new = np.random.default_rng(seed)

    def calculate_acceleration_mask_3D(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
        shape,
        num_offsets,
        seed,
    ) -> np.ndarray:
        undersampling_masks = np.zeros((num_offsets, *shape[1:-1]))
        for offset in range(num_offsets):
            pattern = np.zeros(num_offsets)
            pattern[offset] = 1
            undersampling_masks[offset] = np.tile(pattern, int(np.prod(shape[1:-1]) / num_offsets)).reshape(*shape[1:-1])
        undersampling_masks[:, int(shape[1] * 2/6):int(shape[1] * 4/6), int(shape[2] * 2/6):int(shape[2] * 4/6)] = 1
        return undersampling_masks[None]
    

class MagicMaskFunc(MaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFunc``, this mask will usually undereshoot the
    target acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        if offset % 2 == 0:
            offset_pos = offset + 1
            offset_neg = offset + 2
        else:
            offset_pos = offset - 1 + 3
            offset_neg = offset - 1 + 0

        poslen = (num_cols + 1) // 2
        neglen = num_cols - (num_cols + 1) // 2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        return np.fft.fftshift(mask)  # shift mask and return


class MagicMaskFractionFunc(MagicMaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFractionFunc``, this method exactly matches
    the target acceleration by adjusting the offsets.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_frequencies = round(num_cols * fraction_low_freqs)

        # bound the number of low frequencies between 1 and target columns
        target_columns_to_sample = round(num_cols / acceleration)
        num_low_frequencies = max(min(num_low_frequencies, target_columns_to_sample), 1)

        # adjust acceleration rate based on target acceleration.
        adjusted_target_columns_to_sample = (
            target_columns_to_sample - num_low_frequencies
        )
        adjusted_acceleration = 0
        if adjusted_target_columns_to_sample > 0:
            adjusted_acceleration = round(num_cols / adjusted_target_columns_to_sample)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        accel_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, adjusted_acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, accel_mask, num_low_frequencies


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction_3d":
        return EquispacedMaskFractionFunc3D(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction_dense_center_3d":
        return EquispacedMaskFractionFuncCenterDense3D(center_fractions, accelerations)    
    elif mask_type_str == "equispaced_fraction_full_center_3d":
        return EquispacedMaskFractionFuncCenterFull3D(center_fractions, accelerations)    
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "poisson_3d":
        return PoissonDensitiyMask3D(accelerations)
    elif mask_type_str == "variabledensity3d":
        return VariableDensitiyMask3D(accelerations)
    elif mask_type_str == "cartesian_offset3d":
        return CartesianOffsetMask3D(accelerations)
    else:
        raise ValueError(f"{mask_type_str} not supported")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    vdm = EquispacedMaskFractionFuncCenterDense3D([0], [6])
    # masks = []
    # for _ in tqdm(range(800)):
    #     masks.append(vdm([1, 20, 128, 1]))
    # masks = masks
    masks = [vdm([1, 20, 128, 1])[0] for _ in range(8)]
    masks = torch.stack(masks, 0)
    print(f"Acceleration: {masks[0].numel() / masks[0].sum()}")
    for k in range(8):
        plt.subplot(4, 2, k + 1)
        plt.imshow(masks[k].squeeze())
    plt.show()
    plt.figure()
    plt.imshow(masks.sum(0))
    plt.show()

    # def create_mask(shape, num_offsets=16):
    #     undersampling_masks = np.zeros((num_offsets, *shape))
    #     for offset in range(num_offsets):
    #         pattern = np.zeros(num_offsets)
    #         pattern[offset] = 1
    #         undersampling_masks[offset] = np.tile(pattern, int(np.prod(shape) / num_offsets)).reshape(*shape)
    #     undersampling_masks[:, int(shape[0] * 2/6):int(shape[0] * 4/6), int(shape[1] * 2/6):int(shape[1] * 4/6)] = 1
    #     return undersampling_masks
    
    # masks = create_mask((20, 128), 16)
    # print(np.prod(masks.shape) / masks.sum())
    # plt.imshow(masks[0])
    # plt.show()
    