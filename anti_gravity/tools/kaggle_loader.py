import kagglehub
import pandas as pd
import os

def download_kaggle_dataset(dataset_path: str) -> str:
    """
    Download a Kaggle dataset using kagglehub
    
    Args:
        dataset_path: Kaggle dataset path (e.g., "crisyang777/peronsa-e-personality-shaped-emotion-dataset")
    
    Returns:
        Local path to downloaded dataset
    """
    print(f"📥 Downloading dataset: {dataset_path}")
    path = kagglehub.dataset_download(dataset_path)
    print(f"   ✅ Downloaded to: {path}")
    return path

def load_kaggle_dataset(dataset_path: str, file_pattern: str = ".csv") -> pd.DataFrame:
    """
    Load a Kaggle dataset from downloaded path
    
    Args:
        dataset_path: Path returned by download_kaggle_dataset
        file_pattern: Pattern to match files (default: ".csv")
    
    Returns:
        DataFrame containing the dataset
    """
    # Find all CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(file_pattern):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise ValueError(f"No {file_pattern} files found in {dataset_path}")
    
    # Load the first CSV file (or merge if multiple)
    if len(csv_files) == 1:
        print(f"📖 Loading file: {os.path.basename(csv_files[0])}")
        df = pd.read_csv(csv_files[0])
    else:
        print(f"📖 Found {len(csv_files)} CSV files. Loading first one: {os.path.basename(csv_files[0])}")
        df = pd.read_csv(csv_files[0])
    
    print(f"   ✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def get_personality_emotion_dataset() -> pd.DataFrame:
    """
    Specifically load the Peronsa-E Personality-Shaped Emotion Dataset
    """
    dataset_path = "crisyang777/peronsa-e-personality-shaped-emotion-dataset"
    local_path = download_kaggle_dataset(dataset_path)
    df = load_kaggle_dataset(local_path)
    return df
