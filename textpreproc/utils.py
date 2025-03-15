import hashlib
import json
from dataclasses import asdict

from .model import CleaningConfig

def generate_cache_key(text: str, config: CleaningConfig) -> str:
    """generate cache key based on text and cleaning config
    
    Args:
        text (str): input text
        config (CleaningConfig): cleaning config
    
    Returns:
        str: cache key
    """

    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    config_dict = asdict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

    return f"{text_hash}_{config_hash}"