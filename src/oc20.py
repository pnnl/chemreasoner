import torch
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar
import numpy as np
import lzma
import ase
from ase.io import iread
from ase.db import connect
from activate import data  
import glob
import sys


urls = {'200k': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar',
        '2M': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar',
        'val_id': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar',
        'val_ood_ads_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_ads_trajectories.tar',
        'val_ood_cat_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_cat_trajectories.tar',
        'val_ood_both_traj': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_val_ood_both_trajectories.tar',
       }

subdirs = {'200k': 's2ef_train_200K/s2ef_train_200K',
           '2M': 's2ef_train_2M/s2ef_train_2M',
           'val_id': 's2ef_val_id/s2ef_val_id',
           'val_ood_ads_traj': 'is2res_val_ood_ads_trajectories',
           'val_ood_cat_traj': 'is2res_val_ood_cat_trajectories',
           'val_ood_both_traj': 'is2res_val_ood_both_trajectories',
          }


class OC20(InMemoryDataset):
    r"""
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """ 

    def __init__(self, root: str, tag: str = '200k', n_preds: int = -1, total_energy: bool = False):
        # set up tag
        self.total_energy = total_energy
        self.tag = tag
        if tag not in urls.keys():
            sys.exit(f"tag={tag} is not valid.")


        
        self.download_url = urls[tag]
        self.raw_subdir = subdirs[tag]

        if not total_energy:
            self.tag += '-ref'

        
        super().__init__(root)
        self.target = 0
        self.data, self.slices = torch.load(self.processed_paths[0])


        if not os.path.isfile(os.path.join(self.processed_dir,f'split-{self.tag}.npz')):
            self._generate_split(n_preds=n_preds)

    def atomref(self):
        return None

    def interaction_graph(self, cutoff, **kwargs):
        return data.PBCInteractionGraph(cutoff=cutoff)
        
    @property
    def raw_file_names(self) -> List[str]:
        return [osp.join(self.raw_subdir,'*.extxyz.xz')]

    @property
    def processed_file_names(self) -> str:
        return f'OC20-{self.tag}.pt'

    def NO_download(self):
        if not osp.isdir(osp.join(self.root, self.raw_subdir)):  
            path = download_url(self.download_url, self.raw_dir)
            extract_tar(path, self.raw_dir, mode='r')
            os.unlink(path)

    def _extract_properties(self, mol, i, ref):
        
        #rcell, _ = ase.geometry.minkowski_reduce(ase.geometry.complete_cell(mol.cell), pbc=mol.pbc)
        energy = mol.get_potential_energy()

        if self.total_energy:
            y = energy
        else:
            y = energy-ref['reference_energy']
            

        # set fixed atoms for relaxation
        fixed = np.zeros(len(mol))
        fixed[mol.constraints[0].todict()['kwargs']['indices']]=1
        
        data = Data(
            atomic_numbers=torch.IntTensor(mol.get_atomic_numbers()),
            pos=torch.Tensor(mol.get_positions()),
            y=torch.Tensor([y]),
            f=torch.Tensor(mol.get_forces()),
            cell=torch.Tensor(np.array(mol.cell))[None,...],
            tags=torch.IntTensor(mol.get_tags()),
            fixed=torch.IntTensor(fixed),
            e_total=torch.Tensor([energy]),
            free_energy=torch.Tensor([mol.info['free_energy']]),
            e_ref=torch.Tensor([ref['reference_energy']]),
            #rcell=torch.Tensor(np.array(rcell)),
            name=mol.symbols.get_chemical_formula(),
            sid=ref['system_id'],
            system_id=ref['system_id'],
            fid=torch.IntTensor([i]),
            idx=torch.IntTensor([i]),
            natoms=torch.IntTensor([len(mol)]),
            pbc=torch.Tensor(mol.pbc)
        )

        return data

    def _read_reference_energy(self, txt_path):
        with lzma.open(txt_path, mode='rt', encoding='utf-8') as fid:
            system_id, frame_number, reference_energy = [],[],[]
            for line in fid:
                system_id += [line.split(',')[0]]
                frame_number += [line.split(',')[1]]
                reference_energy += [line.split(',')[2]]
        df = pd.DataFrame({'system_id':system_id, 'frame_number':frame_number, 'reference_energy':reference_energy})
        df['reference_energy']=pd.to_numeric(df['reference_energy'])
        return df

    def _read_reference_energy_traj(self, txt_path):
        df=pd.read_csv(txt_path, names=['system_id','reference_energy'], dtype={'system_id':'str', 'reference_energy':'float32'})

        return df

    def _process_traj(self):
        # read in .xyz files and get properties

        ref_energy = self._read_reference_energy_traj(os.path.join(self.raw_dir, self.raw_subdir ,'system.txt'))

        m = 0
        data_list = []        
        for p in tqdm(self.xyz_paths):
            system_id = p.split('/')[-1].replace('.extxyz.xz','')
            refE = ref_energy.loc[ref_energy['system_id']==system_id].iloc[0]
            # get properties
            for r, mol in enumerate(iread(p, index=":")):
                data_list.append(self._extract_properties(mol, m, refE))
                m+=1
        
        # collate and save as .pt file
        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def _process_sets(self):
        # read in .xyz files and get properties
        m = 0
        data_list = []
        for p in tqdm(self.xyz_paths):
            ref_energy = self._read_reference_energy(p.replace('.extxyz.xz','.txt.xz'))
            
            # get properties
            for r, mol in enumerate(iread(p, index=":")):
                data_list.append(self._extract_properties(mol, m, ref_energy.iloc[r]))
                m+=1
        
        # collate and save as .pt file
        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def process(self):
        # check if data has already been downloaded
        #self._download()
        
        self.xyz_paths = glob.glob(self.raw_paths[0])

        if 'traj' in self.tag:
            self._process_traj()
        else:
            self._process_sets()
            

    def _generate_split(self, n_preds=-1):
        # arbitrary 80:10:10 train:val:test split
        # pred_idx == test_idx
        data = torch.load(self.processed_paths[0])
        n_samples = len(data[0]['idx'])

        indices = np.arange(0, n_samples, 1, dtype=int)

        if n_preds == -1:
            pred_indices = indices.copy()
        else:
            pred_indices = np.random.choice(indices, size=n_preds, replace=False)
        
        np.random.shuffle(indices)

        n_train = int(np.floor(n_samples*0.8))
        n_val = int(np.floor(n_samples*0.1))

        np.savez(file=os.path.join(self.processed_dir,f'split-{self.tag}.npz'),
                 train_idx=indices[:n_train],
                 val_idx=indices[n_train:n_train+n_val],
                 test_idx=indices[n_train+n_val:],
                 pred_idx=pred_indices,
                 mean=data[0]['y'].mean().numpy(),
                 stddev=data[0]['y'].std().numpy(),
                )

