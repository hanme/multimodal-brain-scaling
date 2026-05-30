from pathlib import Path
import hashlib
import yaml


def deep_update(d, u):
    """
    Recursively updates a dictionary `d` with another dictionary `u`.
    If the value of a key in `u` is a dictionary, the function will recursively update
    the corresponding dictionary in `d`. Otherwise, it will directly update the value
    in `d` with the value from `u`.
    Parameters:
    d (dict): The dictionary to be updated.
    u (dict): The dictionary with updates.
    Returns:
    dict: The updated dictionary `d`.
    """

    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_yaml(yaml_file: str) -> dict:
    """
    Loads a yaml file and returns the content as a dictionary.

    Parameters:
    yaml_file (str): The path to the yaml file to be loaded.

    Returns:
    dict: The content of the yaml file as a dictionary.
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    base_config = config.get("base_config", None)
    if base_config is not None:
        yaml_file = Path(yaml_file)
        base_config = yaml_file.parent / base_config
        # Load a base configuration files if exists
        defaults = load_yaml(base_config)
        # Update the defaults with the current configuration
        config = deep_update(defaults, config)

    return config


def get_md5_hash(data: dict) -> str:
    """
    Calculates the MD5 hash of the input data.
    """
    data_str = str(data)
    return hashlib.md5(data_str.encode()).hexdigest()
