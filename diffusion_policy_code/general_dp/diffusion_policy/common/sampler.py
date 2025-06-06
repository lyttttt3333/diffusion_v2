import os
from typing import Optional
from cycler import K
import numpy as np
import numba
import h5py
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray, sequence_length: int,
        episode_mask: np.ndarray,
        pad_before: int = 0, pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length,
                                 episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx -
                        sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(
            len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=dict(),
                 key_last_k=dict(),
                 episode_mask: Optional[np.ndarray] = None,
                 dataset_dir: Optional[str] = None,
                 d3fields_feats_type: Optional[str] = 'no_feats',
                 shape_meta: Optional[dict] = None,
                 ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert (sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends,
                                     sequence_length=sequence_length,
                                     pad_before=pad_before,
                                     pad_after=pad_after,
                                     episode_mask=episode_mask
                                     )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.key_last_k = key_last_k
        self.episode_ends = episode_ends
        self.dataset_dir = dataset_dir
        self.d3fields_feats_type = d3fields_feats_type
        self.shape_meta = shape_meta

    def __len__(self):
        return len(self.indices)

    def idx_to_epi_idx(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        epi_idx = np.searchsorted(
            self.episode_ends, buffer_start_idx, side='right')
        epi_offset = buffer_start_idx - \
            self.episode_ends[epi_idx - 1] if epi_idx > 0 else buffer_start_idx
        return epi_idx, epi_offset

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        is_joint = 'key' in self.shape_meta['action'] and self.shape_meta['action']['key'] == 'joint_action'
        if 'd3fields' in self.shape_meta['obs']:
            use_seg = self.shape_meta['obs']['d3fields']['info'][
                'use_seg'] if 'use_seg' in self.shape_meta['obs']['d3fields']['info'] else True
            feats_prefix = '' if use_seg else '_no_seg'
            if is_joint:
                feats_prefix += '_joint'

        if self.d3fields_feats_type == 'full':
            feats_folder = f'feats{feats_prefix}'
        elif self.d3fields_feats_type == 'pca':
            feats_folder = f'pca_feats{feats_prefix}'
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k and key not in self.key_last_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
                if key == 'd3fields' and (not self.d3fields_feats_type == 'no_feats'):
                    epi_idx, epi_offset = self.idx_to_epi_idx(idx)
                    with h5py.File(os.path.join(self.dataset_dir, feats_folder, f'episode_{epi_idx}.hdf5'), 'r') as f:
                        feats = f['feats'][epi_offset:epi_offset +
                                           sample.shape[0]]
                    sample = np.concatenate([sample, feats], axis=-1)
            elif key in self.key_last_k:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_last_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                sample[-k_data:] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                # sample[-k_data:] = input_arr[buffer_end_idx-k_data:buffer_end_idx]
            elif key in self.key_first_k:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                if key == 'd3fields' and (not self.d3fields_feats_type == 'no_feats'):
                    epi_idx, epi_offset = self.idx_to_epi_idx(idx)
                    with h5py.File(os.path.join(self.dataset_dir, feats_folder, f'episode_{epi_idx}.hdf5'), 'r') as f:
                        feats = f['feats'][epi_offset:epi_offset+k_data]
                        feats_pad = np.full((n_data,) + feats.shape[1:],
                                            fill_value=np.nan, dtype=feats.dtype)
                        if feats.shape[0] < k_data:
                            raise ValueError(
                                f'feats.shape[0] < k_data: {feats.shape[0]} < {k_data}')
                        feats_pad[:k_data] = feats
                    if feats_pad.shape[1] < sample.shape[1]:
                        feats_pad = np.concatenate([feats_pad, np.zeros(
                            (feats_pad.shape[0], sample.shape[1]-feats_pad.shape[1], feats_pad.shape[-1]))], axis=1)
                    elif feats_pad.shape[1] > sample.shape[1]:
                        feats_pad = feats_pad[:, :sample.shape[1]]
                    sample = np.concatenate([sample, feats_pad], axis=-1)
            if key == 'd3fields':
                n_pts = self.shape_meta['obs']['d3fields']['shape'][1]
                sample = sample[:, :n_pts]
            data = sample
            is_pad = np.zeros(self.sequence_length, dtype=bool)
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + data.shape[1:],
                    dtype=data.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                    is_pad[:sample_start_idx] = True
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                    is_pad[sample_end_idx:] = True
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
            result[f'{key}_is_pad'] = is_pad
        return result
