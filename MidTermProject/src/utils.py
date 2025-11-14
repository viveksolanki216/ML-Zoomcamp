import yaml
from pathlib import Path

def get_config_params(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    input_data_path = config.get("input_data")
    return input_data_path