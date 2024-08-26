import torch
import torch.nn as nn
class Utils():
    def count_new_mean(self, prevMean, prevLen, newData):
        return ((prevMean * prevLen) + newData.sum(0)) / (prevLen + newData.shape[0])
      
    def count_new_std(self, prevStd, prevLen, newData):
        return (((prevStd.pow(2) * prevLen) + (newData.var(0) * newData.shape[0])) / (prevLen + newData.shape[0])).sqrt()

    def normalize(self, data, mean = None, std = None, clip = None):
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            data_normalized = (data - mean) / (std + 1e-8)            
        else:
            data_normalized = (data - data.mean()) / (data.std() + 1e-8)
                    
        if clip:
            data_normalized = torch.clamp(data_normalized, -1 * clip, clip)

        return data_normalized