from __future__ import print_function, division
from typing import Optional, List, Union
import numpy as np
import json
import random
import torch
import torch.utils.data
from scipy.ndimage import zoom
from . import VolumeDataset
from ..utils import reduce_label, tile2volume
class TileDataset(torch.utils.data.Dataset):
    def __init__(self,
                 chunk_num: List[int] = [2, 2, 2],
                 chunk_ind: Optional[list] = None,
                 chunk_ind_split: Optional[Union[List[int], str]] = None,
                 chunk_iter: int = -1,
                 chunk_stride: bool = True,
                 volume_json: List[str] = ['path/to/image.json'],
                 label_json: Optional[List[str]] = None,
                 valid_mask_json: Optional[List[str]] = None,
                 mode: str = 'train',
                 pad_size: List[int] = [0, 0, 0],
                 data_scale: List[float] = [1.0, 1.0, 1.0],
                 coord_range: Optional[List[List[int]]] = None,
                 do_relabel: bool = True,
                 **kwargs):
        self.kwargs = kwargs
        self.mode = mode
        self.chunk_iter = chunk_iter
        self.pad_size = pad_size
        self.data_scale = data_scale
        self.do_relabel = do_relabel
        self.chunk_step = 1
        if chunk_stride and self.mode == 'train':  
            self.chunk_step = 2
        self.chunk_num = chunk_num
        self.chunk_ind = self.get_chunk_ind(chunk_ind, chunk_ind_split)
        self.chunk_id_done = []
        self.num_volumes = len(volume_json) 
        if self.mode == 'test':
            assert self.num_volumes == 1, "Only one json file should be given in inference!"
        self.json_volume = [json.load(open(volume_json[i])) for i in range(self.num_volumes)]
        self.json_label = [json.load(open(label_json[i])) for i in range(self.num_volumes)] if (
            label_json is not None) else None
        self.json_valid = [json.load(open(valid_mask_json[i])) for i in range(self.num_volumes)] if (
            valid_mask_json is not None) else None
        self.json_size = [[
            self.json_volume[i]['depth'],
            self.json_volume[i]['height'],
            self.json_volume[i]['width']]
            for i in range(self.num_volumes)] 
        self.coord_m = np.array([[
            0, self.json_volume[i]['depth'],
            0, self.json_volume[i]['height'],
            0, self.json_volume[i]['width']]
            for i in range(self.num_volumes)], int)
        self.get_coord_range(coord_range)
    def get_chunk_ind(self, chunk_ind, split_rule):
        if chunk_ind is None:
            chunk_ind = list(
                range(np.prod(self.chunk_num)))
        if split_rule is not None:
            if isinstance(split_rule, str):
                split_rule = split_rule.split('-')
            assert len(split_rule) == 2
            rank, world_size = split_rule
            rank, world_size = int(rank), int(world_size)
            assert rank < world_size 
            x = len(chunk_ind) // world_size
            low, high = rank * x, (rank + 1) * x
            if rank == world_size - 1:
                high = len(chunk_ind)
            chunk_ind = chunk_ind[low: high]
        return chunk_ind
    def get_coord_name(self):
        r
        assert self.mode == 'test' and len(self.coord) == 1
        return '-'.join([str(x) for x in self.coord[0]])
    def get_coord_range(self, coord_range):
        if coord_range is not None:
            if isinstance(coord_range[0], int):
                assert len(coord_range) == 6
                self.coord_range = [coord_range for _ in range(self.num_volumes)]
            elif isinstance(coord_range[0], list):
                self.coord_range = coord_range
        else:
            self.coord_range = self.coord_m 
        assert len(self.coord_range) == self.num_volumes
        self.coord_range_l, self.coord_range_r = [], []
        for temp in self.coord_range: 
            self.coord_range_l.append([temp[2*i] for i in range(3)])
            self.coord_range_r.append([temp[2*i+1] for i in range(3)])
    def get_range_axis(self, axis_id, vol_id, axis: str='z'):
        axis_map = {'z': 0, 'y': 1, 'x': 2}
        l_bd = self.coord_range_l[vol_id][axis_map[axis]]
        r_bd = self.coord_range_r[vol_id][axis_map[axis]]
        assert r_bd > l_bd
        length = r_bd - l_bd
        steps = np.array([axis_id, axis_id + self.chunk_step])
        axis_range = np.floor(steps / (
            self.chunk_num[axis_map[axis]] + self.chunk_step-1) * length).astype(int)
        return axis_range + l_bd
    def updatechunk(self, do_load=True):
        r
        if len(self.chunk_id_done) == len(self.chunk_ind):
            self.chunk_id_done = []
        id_rest = list(set(self.chunk_ind)-set(self.chunk_id_done))
        if self.mode == 'train':
            id_sample = id_rest[int(np.floor(random.random()*len(id_rest)))]
        elif self.mode == 'test':
            id_sample = id_rest[0]
        self.chunk_id_done += [id_sample]
        zid = float(id_sample//(self.chunk_num[1]*self.chunk_num[2]))
        yid = float((id_sample//self.chunk_num[2]) % (self.chunk_num[1]))
        xid = float(id_sample % self.chunk_num[2])
        self.coord = []
        for i in range(self.num_volumes):
            z0, z1 = self.get_range_axis(zid, vol_id=i, axis='z')
            y0, y1 = self.get_range_axis(yid, vol_id=i, axis='y')
            x0, x1 = self.get_range_axis(xid, vol_id=i, axis='x')
            self.coord.append(np.array([z0, z1, y0, y1, x0, x1], int))
        if do_load:
            self.loadchunk()
    def loadchunk(self):
        r
        padding = np.array([
            -self.pad_size[0], self.pad_size[0],
            -self.pad_size[1], self.pad_size[1],
            -self.pad_size[2], self.pad_size[2]])
        coord_p = [self.coord[i] + padding for i in range(self.num_volumes)]
        print('load chunk: ', coord_p)
        volume = [
            tile2volume(self.json_volume[i]['image'], coord_p[i], self.coord_m[i],
            tile_sz=self.json_volume[i]['tile_size'], tile_st=self.json_volume[i]['tile_st'],
            tile_ratio=self.json_volume[i]['tile_ratio']) for i in range(self.num_volumes)
        ]
        volume = self.maybe_scale(volume, order=3)
        label = None
        if self.json_label is not None:
            dt = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64}
            label = [
                tile2volume(self.json_label[i]['image'], coord_p[i], self.coord_m[i],
                            tile_sz=self.json_label[i]['tile_size'], tile_st=self.json_label[i]['tile_st'],
                            tile_ratio=self.json_label[i]['tile_ratio'], dt=dt[self.json_label[i]['dtype']],
                            do_im=False) for i in range(self.num_volumes)
            ]
            if self.do_relabel:
                label = [reduce_label(x, do_type=True) for x in label]
            label = self.maybe_scale(label, order=0)
        valid_mask = None
        if self.json_valid is not None:
            valid_mask = [
                tile2volume(self.json_valid[i]['image'], coord_p[i], self.coord_m[i],
                            tile_sz=self.json_valid[i]['tile_size'], tile_st=self.json_valid[i]['tile_st'],
                            tile_ratio=self.json_valid[i]['tile_ratio'], do_im=False) for i in range(self.num_volumes)
            ]
            valid_mask = self.maybe_scale(valid_mask, order=0)
        self.dataset = VolumeDataset(volume, label, valid_mask, mode=self.mode, do_relabel=self.do_relabel,
                                     iter_num=self.chunk_iter if self.mode == 'train' else -1,
                                     **self.kwargs)
    def maybe_scale(self, data, order=0):
        if (np.array(self.data_scale) != 1).any():
            for i in range(len(data)):
                dt = data[i].dtype
                data[i] = zoom(data[i], self.data_scale,
                               order=order).astype(dt)
        return data
