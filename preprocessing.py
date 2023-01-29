from tqdm.notebook import tqdm
import torch
import pandas as pd

def minmax_scale(X, data_range=(0,1)):
    range_min, range_max = data_range
    w, h = X.shape
    X = X.reshape((1,-1))
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (range_max - range_min) + range_min
    return X_scaled.reshape(w,h)

def diff_ndvi_sar_vh(tile, ndvi_idx, vh_idx):
    ndvi = minmax_scale(tile[ndvi_idx].clamp(0))
    vh = minmax_scale(tile[vh_idx])
    return ndvi - vh

def calc_frac_over_thresh(img, thresh=0.5):
    total_vals = 256*256.
    count_bad = img[torch.abs(img)>thresh].shape[0]
    count_bad += torch.isnan(img).sum()
    return round((count_bad/total_vals).item(), 3)

def get_cloud_cover_percent(tile, cloud_mask_idx):
    flat_values = tile[cloud_mask_idx].reshape(-1)
    cloud_pixels_percent = (((flat_values <= 1) & (flat_values >=0.6)).sum().item()) / len(flat_values)
    # nan_pixels = sum(flat_values>1)
    return cloud_pixels_percent

def get_biggest_patch(tile, nvdi_idx):
    """Returns the biggest percentage of pixels which have the same value"""
    unique_values_counts = tile[nvdi_idx].reshape(-1).unique(return_counts=True)[1]
    biggest_patch_percent = (torch.max(unique_values_counts) / 256**2).item()
    return biggest_patch_percent

def calc_quality_scores(dataset):
    scores = []
    for ix, sample in tqdm(enumerate(dataset), total=len(dataset)):
        chipid, month_idx = dataset.df_tile_list.iloc[ix].values
        tile = sample['image'].detach().clone()
        diff_img = diff_ndvi_sar_vh(tile, ndvi_idx=15, vh_idx=12)
        score = 1 - calc_frac_over_thresh(diff_img, thresh=0.5)
        cloud_percent = get_cloud_cover_percent(tile, cloud_mask_idx=10)
        biggest_patch_percent = get_biggest_patch(tile, nvdi_idx=15)
        scores.append((chipid, month_idx, score, cloud_percent, biggest_patch_percent))
    return pd.DataFrame(scores, columns=['chipid', 'month', 'score', 'cloud_ratio', 'biggest_patch_ratio'])