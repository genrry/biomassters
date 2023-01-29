import torch
import torch.nn as nn
from skimage.feature import graycomatrix
from skimage import util, exposure
from torchgeo.transforms import indices
import numpy as np

_EPSILON = 1e-10

def _rescale(X, min_value, max_value):

        X_std = (X - min_value) / (max_value - min_value)
        X = X_std * (1 - 0) + 0
        # X[band] = X[band].clamp(0., 1.)
        return X

class AGBMLog1PScale(nn.Module):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        inputs['label'] = torch.log1p(inputs['label'])
        return inputs


class ClampAGBM(nn.Module):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, vmin=0., vmax=500.) -> None:
        """Initialize ClampAGBM
        Args:
            vmin (float): minimum clamp value
            vmax (float): maximum clamp value, 500 is reasonable default per empirical analysis of AGBM data
        """
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, inputs):
        inputs['label'] = torch.clamp(inputs['label'], min=self.vmin, max=self.vmax)
        return inputs


class DropBands(nn.Module):
    """Drop specified bands by index"""

    def __init__(self, device, bands_to_keep=None) -> None:
        super().__init__()
        self.device = device
        self.bands_to_keep = bands_to_keep

    def forward(self, inputs):
        if not self.bands_to_keep:
            return inputs
        X = inputs['image'].detach()
        if X.ndim == 4:
            slice_dim = 1
        else:
            slice_dim = 0
        inputs['image'] = X.index_select(slice_dim,
                                         torch.tensor(self.bands_to_keep,
                                                      device=self.device
                                                      )
                                         )
        return inputs


class Sentinel2Scale(nn.Module):
    """Scale Sentinel 2 optical channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        scale_val = 9000.  # True scaling is [0, 10000], most info is in [0, 9000] range
        X = X / scale_val

        # CLP values in band 10 are scaled differently than optical bands, [0, 100]
        if X.ndim == 4:
            X[:][10] = X[:][10] * scale_val/100.
        else:
            X[10] = X[10] * scale_val/100.
        return X.clamp(0, 1.)


class Sentinel1Scale(nn.Module):
    """Scale Sentinel 1 SAR channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        s1_max = 10.  # S1 db values range mostly from -50 to +10 per empirical analysis
        s1_min = -50.
        X = (X - s1_min) / (s1_max - s1_min)
        return X.clamp(0, 1)


class AppendRatioAB(nn.Module):
    """Append the ratio of specified bands to the tensor.
    """

    def __init__(self, index_a, index_b):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_ratio(self, band_a, band_b):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        return _rescale(band_a/(band_b + _EPSILON), 0, 4).clamp(0,1)

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        ratio = self._compute_ratio(
            band_a=X[..., self.index_a, :, :],
            band_b=X[..., self.index_b, :, :],
        )
        ratio = ratio.unsqueeze(self.dim)
        sample['image'] = torch.cat([X, ratio], dim=self.dim)
        return sample

class AppendVarGLCM(nn.Module):
    """Append the ratio of specified bands to the tensor.
    """

    def __init__(self, index):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index = index

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        band_scaled  = exposure.rescale_intensity(X[..., self.index, :, :].cpu().numpy(), out_range=(0, 1))
        band_scaled = util.img_as_ubyte(band_scaled)
        variance_glcm = np.var(graycomatrix(band_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256),axis=3)[...,0]
        sample['image'] = torch.cat([X, torch.from_numpy(variance_glcm[None,...]).cuda()], dim=0).type(torch.float32)
        return sample

class AppendCCCI(nn.Module):
    """Append the ratio of specified bands to the tensor.
    """

    def __init__(self, index_nir, index_rededge, index_red):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.index_nir = index_nir
        self.index_rededge = index_rededge
        self.index_red = index_red

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        nir  = X[..., self.index_nir, :, :]
        rededge  = X[..., self.index_rededge, :, :]
        red  = X[..., self.index_red, :, :]
        ccci = (nir - rededge / (nir + rededge + _EPSILON)) / (nir - red / (nir + red + _EPSILON) + _EPSILON)
        ccci = _rescale(ccci, -195, 100).clamp(0,1)

        sample['image'] = torch.cat([X, ccci[None,...]], dim=0)
        return sample

class MinMax(nn.Module):
    """Append the ratio of specified bands to the tensor.
    """

    def __init__(self, other, device, new_range=(0,1)):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.device = device
        self.min_values, self.max_values = self._get_bands_ranges(other)
        self.new_min, self.new_max = new_range

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        
        band = -1
        band_min, band_max =  self.min_values[band],  self.max_values[band]
        X_std = (X[band] - band_min) / (band_max - band_min)
        X[band] = X_std * (self.new_max - self.new_min) + self.new_min
        X[band] = X[band].clamp(0., 1.)
        sample['image'] = X
        return sample

    def _get_bands_ranges(self, dataset, sample_size=1000):
        sample_size = min(sample_size, len(dataset))
        c, w, h = dataset[0]['image'].shape
        mins = torch.zeros((c,), device=self.device)
        maxs = torch.zeros((c,), device=self.device)
        for idx in range(sample_size):
            image, _ = dataset[idx].values()
            mins = torch.minimum(image.reshape((c, w * h)).min(1).values, mins)
            maxs = torch.maximum(image.reshape((c, w * h)).max(1).values, maxs)
        return mins, maxs

class AppendEVI(nn.Module):
    """Append the enhanced vegetation index.
    """

    def __init__(self, index_nir, index_red, index_blue):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_blue = index_blue

    def _compute_ratio(self, nir, red, blue):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        G = 2.5
        C1 = 6
        C2 = 7.5
        L = 1
        return (G * ((nir -  red) / (nir + C1 * red - C2 * blue + L))).clamp(-6,4.6)

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        ratio = self._compute_ratio(
            nir=X[..., self.index_nir, :, :],
            red=X[..., self.index_red, :, :],
            blue=X[..., self.index_blue, :, :],
        )
        ratio = _rescale(ratio, min_value=-6, max_value=4.6)
        ratio = ratio.unsqueeze(self.dim)
        sample['image'] = torch.cat([X, ratio], dim=self.dim)
        return sample

class AppendSAVI(nn.Module):
    """Append the enhanced vegetation index.
    """

    def __init__(self, index_nir, index_red):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_nir = index_nir
        self.index_red = index_red

    def _compute_ratio(self, nir, red):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        L = 0.428
        return (((nir - red) / (nir + red + L)) * (1 + L))

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        ratio = self._compute_ratio(
            nir=X[..., self.index_nir, :, :],
            red=X[..., self.index_red, :, :],
        )
        ratio = _rescale(ratio, min_value=-0.4, max_value=0.9).clamp(0,1)
        ratio = ratio.unsqueeze(self.dim)
        sample['image'] = torch.cat([X, ratio], dim=self.dim)
        return sample


class AppendDPRVI(nn.Module):
    """Append the dual polarization vegetation index.
    """

    def __init__(self, index_vh, index_vv):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        
        self.index_vh = index_vh
        self.index_vv = index_vv

    def _compute_ratio(self, vh, vv):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        return (4*vh)/(vv+vh)

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        ratio = self._compute_ratio(
            vh=X[..., self.index_vh, :, :],
            vv=X[..., self.index_vv, :, :],
        )
        ratio = _rescale(ratio, min_value=0, max_value=2).clamp(0,1)
        ratio = ratio.unsqueeze(self.dim)
        sample['image'] = torch.cat([X, ratio], dim=self.dim)
        return sample

class AppendNDVI(indices.AppendNDVI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendNormalizedDifferenceIndex(indices.AppendNormalizedDifferenceIndex):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=0, max_value=0.6).clamp(0,1)
        return sample

class AppendNDBI(indices.AppendNDBI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendNDRE(indices.AppendNDRE):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendNDSI(indices.AppendNDSI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendNDWI(indices.AppendNDWI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendSWI(indices.AppendSWI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendGNDVI(indices.AppendGNDVI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample

class AppendGBNDVI(indices.AppendGBNDVI):

    def forward(self, *args):
        sample = super().forward(*args)
        sample["image"][-1] = _rescale(sample["image"][-1], min_value=-1, max_value=1)
        return sample