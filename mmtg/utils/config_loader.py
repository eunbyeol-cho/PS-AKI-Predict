import os
import json
import pickle
from pathlib import Path

def load_config(study):
    """
    Load hospital-specific configuration from JSON file.
    
    Args:
        study: Name of the hospital (e.g., 'SNUH', 'SNUBH')
    
    Returns:
        dict: Configuration dictionary
    """
    # Get the base directory (workspace root)
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / "configs" / f"{study}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for {study}. Expected at: {config_path}\n"
            f"Please run preprocess.py first to generate the config file."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def load_data_info(data_path):
    """
    Load data information from pickle files (for backward compatibility).
    Now tries JSON first, falls back to pickle if not found.
    
    Args:
        data_path: Path to the data directory
    
    Returns:
        dict: Dictionary with 'info' and other data
    """
    info_dict_path = os.path.join(data_path, "info_dict.pickle")
    config_json_path = os.path.join(data_path, "config.json")
    
    # Try to load from JSON first (new format)
    if os.path.exists(config_json_path):
        import pandas as pd
        with open(config_json_path, 'r') as f:
            config = json.load(f)
        return {
            'info': config,
            'class2raw': pd.read_pickle(os.path.join(data_path, "class2raw.pickle")),
            'znorm': pd.read_csv(os.path.join(data_path, "znorm.csv")),
            'input_data': None,  # Will be loaded separately
            'null_type_data': None  # Will be loaded separately
        }
    
    # Fall back to pickle for backward compatibility
    if os.path.exists(info_dict_path):
        import pandas as pd
        return {
            'info': pd.read_pickle(info_dict_path),
            'class2raw': pd.read_pickle(os.path.join(data_path, "class2raw.pickle")),
            'znorm': pd.read_csv(os.path.join(data_path, "znorm.csv")),
            'input_data': None,
            'null_type_data': None
        }
    
    raise FileNotFoundError(
        f"Neither config.json nor info_dict.pickle found in {data_path}"
    )

