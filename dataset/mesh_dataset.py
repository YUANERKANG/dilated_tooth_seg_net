import json
import os
import pickle
from os.path import join
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.preprocessing import MoveToOriginTransform
from utils.mesh_io import filter_files
from utils.teeth_numbering import colors_to_label, fdi_to_label


def process_mesh(samples: np.ndarray, face_index: np.ndarray, mesh: trimesh.Trimesh, labels: torch.tensor = None):
    """
    Process mesh by extracting features from sampled points and their corresponding face indices.
    
    Args:
        samples: (N, 3) array of sampled points on the mesh surface
        face_index: (N,) array of face indices corresponding to each sample
        mesh: Original trimesh object
        labels: (optional) label tensor for each sample
    
    Returns:
        samples tensor, mesh triangles, vertex normals, face normals, labels
    """
    # Convert samples to tensor
    samples_tensor = torch.from_numpy(samples.copy()).float()
    
    # Extract triangle vertices for each sampled point using face_index
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces[face_index]]).float()
    
    # Extract vertex normals for each triangle
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces[face_index]]).float()
    
    # Extract face normals for each sampled point
    mesh_face_normals = torch.from_numpy(mesh.face_normals[face_index]).float()
    
    if labels is None:
        # Extract labels from face colors using face_index
        face_labels = colors_to_label(mesh.visual.face_colors.copy())
        labels = torch.from_numpy(face_labels[face_index]).long()
    
    return samples_tensor, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels


class Teeth3DSDataset(Dataset):

    def __init__(self, root: str, raw_folder: str = 'raw', processed_folder: str = 'processed_torch',
                 in_memory: bool = False, verbose: bool = True, pre_transform=None, post_transform=None,
                 force_process=False, train_test_split=1, is_train=True):
        self.root = root
        self.processed_folder = processed_folder
        self.raw_folder = raw_folder
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.in_memory = in_memory
        self.in_memory_data = []
        self.verbose = verbose
        self.file_names = []
        self.train_test_split = train_test_split
        self._set_file_index(is_train)
        self.move_to_origin = MoveToOriginTransform()
        Path(join(self.root, self.processed_folder)).mkdir(parents=True, exist_ok=True)
        Path(join(self.root, self.raw_folder)).mkdir(parents=True, exist_ok=True)
        if not self._is_processed() or force_process:
            self._process()
        self.processed_file_names = filter_files(join(self.root, self.processed_folder), 'pt')
        if self.in_memory:
            self._load_in_memory()
        
    def _set_file_index(self, is_train: bool):
        if self.train_test_split == 1:
            split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',
                                                                                         'testing_upper.txt']
        elif self.train_test_split == 2:
            split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
                else ['private-testing-set.txt']
        else:
            raise ValueError(f'train_test_split should be 1 or 2. not {self.train_test_split}')
        for f in split_files:
            with open(f'data/3dteethseg/raw/{f}') as file:
                for l in file:
                    l = f'data_{l.rstrip()}.pt'
                    if os.path.isfile(join(self.root, self.processed_folder, l)):
                        self.file_names.append(l)


    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _loop(self, data):
        if self.verbose:
            return tqdm(data)
        return data
    
    def _sample_mesh(self, mesh, labels, target_count=16000):
        """
        Sample points uniformly from mesh surface using area-weighted probability.
        
        Args:
            mesh: trimesh object
            labels: per-face labels
            target_count: number of points to sample (default: 16000)
        
        Returns:
            samples: (N, 3) array of sampled points
            face_index: (N,) array of face indices
            labels: (N,) array of labels for each sample
        """
        # Sample uniformly from the mesh surface
        samples, face_index = trimesh.sample.sample_surface(mesh, target_count)
        
        # If we got fewer samples than target, resample with replacement
        if len(samples) < target_count:
            # Resample to reach target count
            additional_count = target_count - len(samples)
            additional_samples, additional_face_index = trimesh.sample.sample_surface(mesh, additional_count)
            samples = np.vstack([samples, additional_samples])
            face_index = np.concatenate([face_index, additional_face_index])
        
        # Extract labels for sampled faces
        sampled_labels = labels[face_index]
        
        return samples, face_index, sampled_labels

    def _iterate_mesh_and_labels(self):
        root_mesh_folder = join(self.root, self.raw_folder)
        for root, dirs, files in os.walk(root_mesh_folder):
            for file in files:
                if file.endswith(".obj"):
                    mesh = trimesh.load(join(root, file))
                    with open(join(root, file).replace('.obj', '.json')) as f:
                        data = json.load(f)
                    labels = np.array(data["labels"])
                    labels = labels[mesh.faces]
                    labels = labels[:, 0]
                    labels = fdi_to_label(labels)
                    samples, face_index, sampled_labels = self._sample_mesh(mesh, labels)
                    fn = file.replace('.obj', '')
                    yield samples, face_index, mesh, sampled_labels, fn

    def _is_processed(self):
        files_processed = filter_files(join(self.root, self.processed_folder), 'pt')
        files_raw = filter_files(join(self.root, self.raw_folder), 'obj')
        return len(files_processed) == len(files_raw)

    def _process(self):
        self._log('Processing data')
        for f in filter_files(join(self.root, self.processed_folder), 'pt'):
            os.remove(join(self.root, self.processed_folder, f))
        for samples, face_index, mesh, labels, fn in self._loop(self._iterate_mesh_and_labels()):
            # Center the samples and shift the original mesh
            samples, mesh = self.move_to_origin(samples, mesh)
            # Process mesh to extract features
            data = process_mesh(samples, face_index, mesh, torch.from_numpy(labels).long())
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            with open(f'{join(self.root, self.processed_folder)}/data_{fn}.pt', 'wb') as f:
                pickle.dump(data, f)
        self._log('Processing done')

    def _load_in_memory(self):
        files_processed = [join(self.root, self.processed_folder, f) for f in self.file_names]
        for i, f in enumerate(files_processed):
            file = open(join(self.root, self.processed_folder, f), 'rb')
            data = pickle.load(file)
            if self.post_transform is not None:
                data = self.post_transform(data)
            self.in_memory_data.append(data)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.in_memory:
            return self.in_memory_data[index]
        else:
            f = self.file_names[index]
            file = open(join(self.root, self.processed_folder, f), 'rb')
            data = pickle.load(file)
            if self.post_transform is not None:
                data = self.post_transform(data)
            return data











