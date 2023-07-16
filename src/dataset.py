import numpy as np
import h5py
import csv
import time
import logging
import logging
import librosa
from . import config


class AmbiDataset(object):
    def __init__(self, hdf5_path):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        Args:
          clip_samples: int
          classes_num: int
        """
        self.hdf5_path = hdf5_path
    
    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        
        Args:
          index: int
        Returns: 
          data_dict: {
            'audio_name': str, 
            'features': (mel_bins, timesteps), 
            'target': (classes_num,)}
        """
        segments = False
        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['info']['audio_name'][index].decode()
            audio_path = hf['info']['audio_path'][index].decode()
            features = hf['features'][str(index)][:]
            target = hf['info']['target'][index].astype(np.float32)
            gt = hf['info']['gt'][index].astype(np.float32)
            #print(features[0].shape)

        data_dict = {
            'audio_name': audio_name, 'audio_path': audio_path, 'features': features, 'target': target, 'gt': gt, 'segments': segments}
         
        return data_dict
    

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hf:
            return len(hf['info']['audio_name'])


def test_collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'feastures': (clip_samples,), ...}, 
                             {'audio_name': str, 'features': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'features': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        if key == 'features':
            np_data_dict[key] = np.concatenate([np.expand_dims(seg, axis=0) for data_dict in list_data_dict for seg in data_dict[key]], axis=0, dtype='float32')
        elif key == 'target' or key == 'gt':
            np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict], dtype='float32')
        else:
            np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict], dtype='object')
    return np_data_dict


def train_collate_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
        #else:
        #    np_data_dict[key] = np.concatenate([data_dict[key] for data_dict in list_data_dict], 
        #    if len(list_data_dict[0][key].shape) > 1:
        #        max_len = max([x[key].shape[-2] for x in list_data_dict])
        #        #print(max_len)
        #        for x in list_data_dict:
        #            new_ = np.pad(x[key], ((0, max_len - len(x[key])), (0, 0)), 'edge')
        #            #print("x key", x[key].shape)
        #            #print(new_)
        #            #print(" padded x", new_.shape)
        #            
        #        np_data_dict[key] = np.concatenate([np.expand_dims(np.pad(x[key], ((0, max_len - len(x[key])), (0, 0)), 'edge'), axis=0) for x in list_data_dict], axis=0) # (timesteps, features)
        #    else:
        #        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
        #        #print("concatenate", np_data_dict[key].shape)
        #if key == 'target' or key == 'gt':
        #    np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict]
        #    if len(list_data_dict[0][key].shape) == 1:
        #        np_data_dict[key] = np.expand_dims(np_data_dict[key], axis=0)
        #else:
        #    
    #print(np_data_dict['features'])
    #print(np_data_dict['features'].shape)

