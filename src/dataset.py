import numpy as np
import h5py

class AmbiDataset(object):
    def __init__(self, hdf5_path: str):
        """
        This class is used by DataLoader. 
        Args:
            hdf5_path: path to stored features (hdf5 path)
        """
        self.hdf5_path = hdf5_path
    
    def __getitem__(self, index: int) -> dict:
        """
        Load waveform and target of an audio clip.
            Args:
                index   : index of the audio clip in the dataset
            Returns: 
                data_dict: Dictionary containing audio features, noisy target, ground truth and soft ground truth for a given audio
        """
        segments = False
        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['info']['audio_name'][index].decode()
            audio_path = hf['info']['audio_path'][index].decode()
            features = hf['features'][str(index)][:]
            target = hf['info']['target'][index].astype(np.float32)
            gt = hf['info']['gt'][index].astype(np.float32)
            soft_gt = hf['info']['soft-gt'][index].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'audio_path': audio_path, 'features': features, 'target': target, 'gt': gt, 'soft-gt': soft_gt, 'segments': segments}
         
        return data_dict
    

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hf:
            return len(hf['info']['audio_name'])


def train_collate_fn(list_data_dict: dict) -> dict:
    """
    Collate data for trainloader.
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
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict


def test_collate_fn(list_data_dict: dict) -> dict:
    """
    Collate data for testloader.
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
        elif key == 'target' or key == 'gt' or key == 'soft-gt':
            np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict], dtype='float32')
        else:
            np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict], dtype='object')
    return np_data_dict

