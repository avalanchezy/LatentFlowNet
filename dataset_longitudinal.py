"""
Dataset for longitudinal PET prediction.

Handles subjects with variable timepoints (T0, T6, T12, T18, T24).
Loads posterior-mean KL-VAE representations for LatentFlowNet.
"""

import os
import re
from typing import Dict

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
import config


def load_nifti(filepath):
    """Load NIfTI file and return numpy array."""
    if not os.path.exists(filepath):
        return None
    data = nib.load(filepath).get_fdata()
    return data.astype(np.float32)


class LongitudinalPETDataset(Dataset):
    """
    Dataset for longitudinal PET prediction with variable timepoints.

    Each subject may have different available timepoints:
    - Always: T0 (baseline), T24 (target)
    - Optional: T6, T12, T18 (intermediate)

    The dataset works with pre-encoded latent representations.
    """

    # Mapping from session names to months
    SESSION_TO_MONTHS = {
        'M00': 0,
        'M06': 6,
        'M12': 12,
        'M18': 18,
        'M24': 24,
    }

    def __init__(
        self,
        data_root,
        subject_list_file=None,
        label_file=None,
        obs_file=None,
        baseline_time=0,
        target_time=24,
        use_latent=True,
        stage='train',
        unknown_label=0,
    ):
        """
        Args:
            data_root: Root directory containing PET data
            subject_list_file: Path to file with subject IDs
            label_file: Path to CSV with subject labels
            baseline_time: Baseline timepoint in months (default: 0)
            target_time: Target timepoint to predict (default: 24)
            use_latent: If True, load from latent directory; else load raw PET
            stage: 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.baseline_time = baseline_time
        self.target_time = target_time
        self.use_latent = use_latent
        self.stage = stage
        self.unknown_label = int(unknown_label)
        self.obs_map: Dict[str, np.ndarray] = {}
        self.obs_dim = 0
        self.default_obs = np.zeros((0,), dtype=np.float32)

        # Load subject list
        if subject_list_file and os.path.exists(subject_list_file):
            with open(subject_list_file, 'r') as f:
                self.subjects = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover subjects from data directory
            self.subjects = self._discover_subjects()

        # Load labels
        self.labels = {}
        if label_file and os.path.exists(label_file):
            df = pd.read_csv(label_file, encoding='ISO-8859-1')
            for _, row in df.iterrows():
                # Support both 'ID' and 'filename' columns
                if 'filename' in df.columns:
                    subject_id = row['filename'].replace('.nii.gz', '').replace('.nii', '')
                elif 'ID' in df.columns:
                    subject_id = row['ID']
                else:
                    continue

                if 'label_id' in df.columns:
                    self.labels[subject_id] = int(row['label_id'])
                elif 'label' in df.columns:
                    self.labels[subject_id] = int(row['label'])

        # Load subject-level metadata
        if obs_file and os.path.exists(obs_file):
            df = pd.read_csv(obs_file, dtype={'subject_id': str})
            if 'subject_id' not in df.columns:
                raise ValueError(f"obs_file must contain subject_id column: {obs_file}")
            obs_cols = [col for col in df.columns if col != 'subject_id']
            self.obs_dim = len(obs_cols)
            self.default_obs = np.zeros((self.obs_dim,), dtype=np.float32)
            for row in df.to_dict(orient='records'):
                subject_id = str(row['subject_id']).strip()
                self.obs_map[subject_id] = np.asarray([row[col] for col in obs_cols], dtype=np.float32)

        # Filter subjects that have both baseline and target
        self.valid_subjects = self._filter_valid_subjects()

        print(f"[{stage}] Found {len(self.valid_subjects)} subjects with T{baseline_time} and T{target_time}")

    def _discover_subjects(self):
        """Auto-discover subjects from data directory."""
        subjects = []
        if os.path.exists(self.data_root):
            for item in os.listdir(self.data_root):
                if os.path.isdir(os.path.join(self.data_root, item)):
                    subjects.append(item)
        return subjects

    def _get_timepoint_path(self, subject_id, month):
        """Get path to PET scan for a subject at a specific timepoint."""
        session = f"M{month:02d}"

        if self.use_latent:
            # Latent representation path
            # Now organized in subdirectories: data_root/subject_id/filename
            filename = f"{subject_id}_ses-{session}.nii.gz"
            return os.path.join(self.data_root, subject_id, filename)
        else:
            # Raw PET path (need to search for correct filename)
            subject_dir = os.path.join(self.data_root, subject_id)
            if not os.path.exists(subject_dir):
                return None

            # Look for files matching this session
            # Robust matching: check for session string in filename (e.g. 'M00' or 'ses-M00')
            for f in os.listdir(subject_dir):
                if f.endswith('.nii.gz') and session in f:
                    return os.path.join(subject_dir, f)

            return None

    def _filter_valid_subjects(self):
        """Filter subjects that have required timepoints."""
        valid = []
        for subject in self.subjects:
            baseline_path = self._get_timepoint_path(subject, self.baseline_time)
            target_path = self._get_timepoint_path(subject, self.target_time)

            if baseline_path and target_path:
                if os.path.exists(baseline_path) and os.path.exists(target_path):
                    valid.append(subject)

        return valid

    def _get_available_intermediates(self, subject_id):
        """Discover scans strictly between the configured endpoint months."""
        subject_dir = os.path.join(self.data_root, subject_id)
        if not os.path.isdir(subject_dir):
            return {}

        intermediates = {}
        for filename in sorted(os.listdir(subject_dir)):
            if not (filename.endswith('.nii.gz') or filename.endswith('.nii')):
                continue
            session_match = re.search(r"ses-M(\d{1,3})", filename)
            if session_match:
                month = int(session_match.group(1))
            else:
                month_matches = re.findall(r"(?:^|[_-])M(\d{1,3})", filename)
                if not month_matches:
                    continue
                month = int(month_matches[-1])
            if not self.baseline_time < month < self.target_time:
                continue
            if month in intermediates:
                raise RuntimeError(f"Multiple M{month:02d} scans found for {subject_id}")
            intermediates[month] = os.path.join(subject_dir, filename)
        return intermediates

    def __len__(self):
        return len(self.valid_subjects)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'z_0': latent at baseline (C, D, H, W)
                - 'z_T': latent at target time (C, D, H, W) - ground truth
                - 'intermediates': dict {month: latent} of available intermediates
                - 'available_times': list of available timepoints
                - 'subject_id': string
                - 'label': disease label (int) or -1 if not available
        """
        subject_id = self.valid_subjects[idx]

        # Load baseline
        baseline_path = self._get_timepoint_path(subject_id, self.baseline_time)
        z_0 = load_nifti(baseline_path)
        if z_0 is None:
            raise RuntimeError(f"Could not load baseline for {subject_id}")

        # Load target
        target_path = self._get_timepoint_path(subject_id, self.target_time)
        z_T = load_nifti(target_path)
        if z_T is None:
            raise RuntimeError(f"Could not load target for {subject_id}")

        # Load intermediates
        intermediate_paths = self._get_available_intermediates(subject_id)
        intermediates = {}
        available_times = [self.baseline_time]

        for month, path in intermediate_paths.items():
            data = load_nifti(path)
            if data is not None:
                # Add channel dimension if needed
                if data.ndim == 3:
                    data = np.expand_dims(data, 0)
                elif data.ndim == 4:
                    data = data.transpose(3, 0, 1, 2)
                intermediates[month] = torch.from_numpy(data)
                available_times.append(month)

        available_times.append(self.target_time)
        available_times.sort()

        # Add channel dimension if needed
        if z_0.ndim == 3:
            z_0 = np.expand_dims(z_0, 0)
        elif z_0.ndim == 4:
            z_0 = z_0.transpose(3, 0, 1, 2)

        if z_T.ndim == 3:
            z_T = np.expand_dims(z_T, 0)
        elif z_T.ndim == 4:
            z_T = z_T.transpose(3, 0, 1, 2)

        # Get label
        label = self.labels.get(subject_id, self.unknown_label)
        obs = self.obs_map.get(subject_id, self.default_obs)
        obs_mask = np.array([1.0 if subject_id in self.obs_map else 0.0], dtype=np.float32)

        return {
            'z_0': torch.from_numpy(z_0),
            'z_T': torch.from_numpy(z_T),
            'intermediates': intermediates,
            'available_times': available_times,
            'subject_id': subject_id,
            'label': label,
            'obs': torch.from_numpy(obs.copy()),
            'obs_mask': torch.from_numpy(obs_mask),
        }


class SimpleLongitudinalDataset(Dataset):
    """
    Simplified dataset for basic T0 -> T24 prediction.

    Works with the ADNI-style data structure where files are named like:
    sub-ADNI{ID}_ses-M{XX}_..._pet.nii.gz
    """

    def __init__(
        self,
        data_root,
        label_file=None,
        stage='train',
        unknown_label=0,
    ):
        """
        Args:
            data_root: Root directory containing subject folders
            label_file: Path to CSV with subject labels
            stage: 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.stage = stage
        self.unknown_label = int(unknown_label)

        # Load labels
        self.labels = {}
        if label_file and os.path.exists(label_file):
            df = pd.read_csv(label_file, encoding='ISO-8859-1')
            for _, row in df.iterrows():
                if 'filename' in df.columns:
                    subject_id = str(row['filename']).replace('.nii.gz', '')
                elif 'ID' in df.columns:
                    subject_id = str(row['ID'])
                else:
                    continue

                label_col = 'label_id' if 'label_id' in df.columns else 'label'
                if label_col in df.columns:
                    self.labels[subject_id] = int(row[label_col])

        # Discover subjects with T0 and T24
        self.samples = self._discover_samples()
        print(f"[{stage}] Found {len(self.samples)} samples with M00 and M24")

    def _discover_samples(self):
        """Find all subjects with both M00 and M24 scans."""
        samples = []

        if not os.path.exists(self.data_root):
            return samples

        for subject_dir in os.listdir(self.data_root):
            subject_path = os.path.join(self.data_root, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # Find M00 and M24 files
            m00_file = None
            m24_file = None
            intermediate_files = {}

            for f in os.listdir(subject_path):
                if not f.endswith('.nii.gz'):
                    continue

                if 'M00' in f:
                    m00_file = os.path.join(subject_path, f)
                elif 'M24' in f:
                    m24_file = os.path.join(subject_path, f)
                elif 'M06' in f:
                    intermediate_files[6] = os.path.join(subject_path, f)
                elif 'M12' in f:
                    intermediate_files[12] = os.path.join(subject_path, f)
                elif 'M18' in f:
                    intermediate_files[18] = os.path.join(subject_path, f)

            if m00_file and m24_file:
                samples.append({
                    'subject_id': subject_dir,
                    'm00_path': m00_file,
                    'm24_path': m24_file,
                    'intermediates': intermediate_files
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load data
        z_0 = load_nifti(sample['m00_path'])
        z_T = load_nifti(sample['m24_path'])

        # Load intermediates
        intermediates = {}
        available_times = [0]
        for month, path in sample['intermediates'].items():
            data = load_nifti(path)
            if data is not None:
                if data.ndim == 3:
                    data = np.expand_dims(data, 0)
                elif data.ndim == 4:
                    data = data.transpose(3, 0, 1, 2)
                intermediates[month] = torch.from_numpy(data)
                available_times.append(month)

        available_times.append(24)
        available_times.sort()

        # Add channel dimension
        if z_0.ndim == 3:
            z_0 = np.expand_dims(z_0, 0)
        elif z_0.ndim == 4:
            z_0 = z_0.transpose(3, 0, 1, 2)

        if z_T.ndim == 3:
            z_T = np.expand_dims(z_T, 0)
        elif z_T.ndim == 4:
            z_T = z_T.transpose(3, 0, 1, 2)

        # Get label
        label = self.labels.get(sample['subject_id'], self.unknown_label)

        return {
            'z_0': torch.from_numpy(z_0),
            'z_T': torch.from_numpy(z_T),
            'intermediates': intermediates,
            'available_times': available_times,
            'subject_id': sample['subject_id'],
            'label': label
        }


def collate_longitudinal(batch):
    """
    Custom collate function for variable-length intermediate timepoints.
    """
    z_0 = torch.stack([b['z_0'] for b in batch])
    z_T = torch.stack([b['z_T'] for b in batch])

    labels = torch.tensor([b['label'] for b in batch])
    subject_ids = [b['subject_id'] for b in batch]
    available_times = [b['available_times'] for b in batch]
    obs = torch.stack([b['obs'] for b in batch])
    obs_mask = torch.stack([b['obs_mask'] for b in batch])

    # Intermediates remain as list of dicts (variable per sample)
    intermediates = [b['intermediates'] for b in batch]

    return {
        'z_0': z_0,
        'z_T': z_T,
        'intermediates': intermediates,
        'available_times': available_times,
        'subject_ids': subject_ids,
        'labels': labels,
        'obs': obs,
        'obs_mask': obs_mask,
    }


# For testing
if __name__ == "__main__":
    # Test with current data structure
    dataset = SimpleLongitudinalDataset(
        data_root="./data",
        stage="test"
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Subject: {sample['subject_id']}")
        print(f"z_0 shape: {sample['z_0'].shape}")
        print(f"z_T shape: {sample['z_T'].shape}")
        print(f"Available times: {sample['available_times']}")
        print(f"Intermediates: {list(sample['intermediates'].keys())}")
        print(f"Label: {sample['label']}")
    else:
        print("No valid samples found!")
