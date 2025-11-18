import json
from func_lm_to_graph import landmarks_to_graph_with_features, skeleton_edges
import os
from glob import glob
from torch.utils.data import Dataset
import torch
from torch_geometric.loader import DataLoader

class FallDetectionDatasetKFold(Dataset):
    def __init__(self, fold_number=1, is_train=True, split_dir='splits', print_path=False):
        """
        Initialize Fall Detection Dataset with K-Fold
        
        Args:
            fold_number (int): Which fold to use (1-based index)
            is_train (bool): If True, load training set, else load validation set
            split_dir (str): Directory containing the split files
            print_path (bool): Whether to print file paths when loading data
        """
        self.print_path = print_path
        
        # Load split data for the specified fold
        split_file = os.path.join(split_dir, f'5fold_split_fold{fold_number}.json')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found. Please generate k-fold splits first.")
        
        print(f"Loading fold {fold_number} dataset from {split_file}")
        
        # Load split data
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Select appropriate file paths based on is_train
        rel_files = split_data['train_files'] if is_train else split_data['val_files']
        
        # Normalize file paths to use forward slashes for cross-platform compatibility
        self.file_paths = [os.path.join(os.getcwd(), f.replace('\\', '/')) for f in rel_files]
        
        if print_path:
            # Print some sample file paths for debugging
            print(f"{'Train' if is_train else 'Validation'} set sample files:")
            for i, path in enumerate(self.file_paths[:5]):
                print(f"  {i+1}: {path}")
            print(f"Total files: {len(self.file_paths)}")
            
            # Count fall and non-fall instances
            fall_count = sum(1 for path in self.file_paths if '/fall/' in path)
            not_fall_count = sum(1 for path in self.file_paths if '/not_fall/' in path)
            print(f"  Fall instances: {fall_count}")
            print(f"  Not-fall instances: {not_fall_count}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Get file path
        file_path = self.file_paths[idx]
        
        if self.print_path:
            print(f"Loading file: {file_path}")
        
        # Extract label from path ('fall' -> 1, 'not_fall' -> 0)
        if '/fall/' in file_path:
            label = 1
        else:
            label = 0
        y = torch.tensor([label], dtype=torch.float).squeeze()
        
        # Load landmarks from JSON
        try:
            with open(file_path, 'r') as f:
                landmarks = json.load(f)
        except FileNotFoundError:
            # If file is not found, try with normalized path
            normalized_path = file_path.replace('\\', '/')
            with open(normalized_path, 'r') as f:
                landmarks = json.load(f)
        
        # Convert landmarks to graph data with engineered features
        graph_data = landmarks_to_graph_with_features(landmarks, y, skeleton_edges)
        
        
        
        return graph_data


if __name__ == "__main__":
    
    # Test loading a dataset
    print("\nTesting dataset loading:")
    dataset = FallDetectionDatasetKFold(fold_number=1, is_train=True, print_path=True)
    
    
    # Get first item
    print("\nTesting first item retrieval:")
    first_item = dataset[0]
    print(f"First item node features: {first_item.x}")
    print(f"First item node features shape: {first_item.x.shape}")
    print(f"First item edge index: {first_item.edge_index}")
    print(f"First item edge index shape: {first_item.edge_index.shape}")
    print(f"First item label: {first_item.y}")
    
    # Test with DataLoader
    print("\nTesting DataLoader:")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Labels: {batch.y}")
        if hasattr(batch, 'engineered'):
            print(f"Engineered features shape: {batch.engineered.shape}")
        break  # Only test first batch