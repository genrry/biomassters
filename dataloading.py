import pandas as pd
import rasterio
import transforms as tf
import numpy as np
import os
from collections import Counter
from torch.utils.data import Dataset
import torch
from glob import glob
from google.cloud import storage
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class SentinelDataset(Dataset):
    '''Sentinel 1 & 2 dataset.'''

    def __init__(self, tile_file, dir_tiles, dir_target,
                 max_chips=None, transform=None, device='cpu', 
                 gcp_bucket_name=True, remove_missing=True, scale=True
                 ):
        '''
        Args:
            tile_file -- path to csv file specifying chipid and month for each tile to be loaded
            dir_tiles -- path to directory containing Sentinel data tiles
            dir_target -- path to directory containing target data (AGWB) tiles
            max_chips -- maximum number of chips to load, used for testing, None --> load all
            transform -- transforms to apply to each sample/batch
            device -- device to load data onto ('cpu', 'mps', 'cuda')
        '''
        self.gcp_bucket_name = gcp_bucket_name
        self.remove_missing = remove_missing
        if  self.gcp_bucket_name:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.gcp_bucket_name)
        if tile_file:
            self.df_tile_list = pd.read_csv(tile_file, index_col=0)[['chipid','month']]
        else:
            if gcp_bucket_name:
                self.df_tile_list = self._make_gcp_df_tile_list(dir_tiles)
            else:
                self.df_tile_list = self._make_df_tile_list(dir_tiles)
            
        if max_chips:
            self.df_tile_list = self.df_tile_list[:max_chips]
        self.dir_tiles = dir_tiles
        self.dir_target = dir_target
        self.device = device
        self.transform_s2 = tf.Sentinel2Scale()
        self.transform_s1 = tf.Sentinel1Scale()
        self.transform = transform
        self.scale_transforms = False
        if scale:
            self.min_max_scaler = tf.MinMax(self, device)
            self.scale_transforms = True

    def __len__(self):
        return len(self.df_tile_list)

    def __getitem__(self, idx):
        chipid, month = self.df_tile_list.iloc[idx].values
        # Sentinel 1
        try:
            s1_tile = self._load_sentinel_tiles('S1', chipid, month)
            s1_tile_scaled = self.transform_s1(s1_tile)
            # s1_tile_scaled = s1_tile
        except:
            # print(f'Data load failure for S1: {chipid} {month}')
            s1_tile_scaled = torch.full([4, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)
        # Sentinel 2
        try:
            s2_tile = self._load_sentinel_tiles('S2', chipid, month)
            # s2_tile_scaled = s2_tile
            s2_tile_scaled = self.transform_s2(s2_tile)
        except:
            # print(f'Data load failure for S2: {chipid} {month}')
            s2_tile_scaled = torch.full([11, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        sentinel_tile = torch.cat([s2_tile_scaled, s1_tile_scaled], axis=0)

        if self.dir_target:
            target_tile = self._load_agbm_tile(chipid)
        else:
            target_tile = torch.full([1, 256, 256], torch.nan, dtype=torch.float32, requires_grad=False, device=self.device)

        sample = {'image': sentinel_tile, 'label': target_tile} # 'image' and 'label' are used by torchgeo

        if self.transform:
            sample = self.transform(sample)
        if self.scale_transforms:
            sample = self.min_max_scaler(sample)

        return sample

    def _read_tif_to_tensor(self, tif_path):
        with rasterio.open(tif_path) as src:
            X = torch.tensor(src.read().astype(np.float32),
                             dtype=torch.float32,
                             device=self.device,
                             requires_grad=False,
                             )
        return X

    def _read_gcp_tif_to_tensor(self, tif_path):
        blob = self.bucket.blob(tif_path)
        with blob.open("rb") as f:
            with rasterio.MemoryFile(f.read()) as memfile:
                    with memfile.open() as src:
                        X = torch.tensor(src.read().astype(np.float32),
                                        dtype=torch.float32,
                                        device=self.device,
                                        requires_grad=False,
                                        )
        return X

    def _load_sentinel_tiles(self, sentinel_type, chipid, month):
        file_name = f'{chipid}_{sentinel_type}_{str(month).zfill(2)}.tif'
        tile_path = os.path.join(self.dir_tiles, file_name)
        if self.gcp_bucket_name:
            return self._read_gcp_tif_to_tensor(tile_path)
        else:
            return self._read_tif_to_tensor(tile_path)
            

    def _load_agbm_tile(self, chipid):
        target_path = os.path.join(self.dir_target,
                                   f'{chipid}_agbm.tif')
        if self.gcp_bucket_name:
            return self._read_gcp_tif_to_tensor(target_path)
        else:
            return self._read_tif_to_tensor(target_path)
        

    def _make_df_tile_list(self, dir_tiles):
        tile_files = [
            os.path.basename(f).split('.')[0] for f in glob(f'{dir_tiles}/*.tif')
        ]
        tile_tuples = []
        for tile_file in tile_files:
            chipid, _, month = tile_file.split('_')
            tile_tuples.append(tuple([chipid, int(month)]))
        if self.remove_missing:
            tile_tuples = [values for values,count in Counter(tile_tuples).most_common() if count>1]
        tile_tuples = list(set(tile_tuples))
        tile_tuples.sort()
        return pd.DataFrame(tile_tuples, columns=['chipid', 'month'])

    def _make_gcp_df_tile_list(self, dir_tiles):
        blobs = self.storage_client.list_blobs(self.gcp_bucket_name, prefix=dir_tiles)
        tile_files = [
            os.path.basename(f).split('.')[0] for f in [blob.name for blob in blobs]
        ]
        tile_tuples = []
        for tile_file in tile_files:
            chipid, _, month = tile_file.split('_')
            tile_tuples.append(tuple([chipid, int(month)]))
        tile_tuples = list(set(tile_tuples))
        tile_tuples.sort()
        return pd.DataFrame(tile_tuples, columns=['chipid', 'month'])