import torch.utils.data as data
import numpy as np
import torch



class PointCloudDataset(data.Dataset):

    def __init__(self,target_files,phase,transform):
        self.transform=transform
        self.target_files=target_files

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self,index):
        pcd,one_hot = self.pull_item(index)
        return pcd,one_hot

    def pull_item(self,index):
        self.path=self.target_files[index]

        #label
        label = int(self.path[2])

        #load_pcd
        pcd = np.load('DATA/MSRAction_npz_pp/' + self.path,allow_pickle=True)['arr_0']
        pcd = torch.from_numpy(pcd.astype(np.float32)).clone()
        #pcd=self.transform(pcd)
        return pcd, label