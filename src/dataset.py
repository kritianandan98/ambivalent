import numpy as np
import h5py
import csv
import time
import logging
import logging


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

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['info']['audio_name'][index].decode()
            features = hf['features'][str(index)][:]
            target = hf['info']['target'][index].astype(np.float32)
            gt = hf['info']['gt'][index].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'features': features, 'target': target, 'gt': gt}
         
        return data_dict
    

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hf:
            return len(hf['info']['audio_name'])


class TestSampler(object):
    def __init__(self, hdf5_path, batch_size):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(hdf5_path, 'r') as hf:
            self.indexes = np.arange(len(hf['audio_name']))
        self.audios_num = len(self.indexes)
        
    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for i in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.indexes[i]})

            pointer += batch_size
            yield batch_meta

    def __len__(self):
        return int(self.audios_num / self.batch_size) + 1


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict

