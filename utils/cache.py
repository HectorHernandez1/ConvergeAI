import json
import hashlib
import time
import os
from typing import Optional
from datetime import datetime, timedelta
from config import settings

CACHE_DIR = os.path.join(settings.log_dir, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(model_name: str, prompt: str) -> str:
    """Generate a cache key from model name and prompt.

    Uses 'surrogatepass' encoding to handle special characters (e.g., mathematical
    symbols like \ud835) that may appear in PDF/PPT extracted text.
    """
    key_string = f"{model_name}:{prompt}"
    return hashlib.sha256(key_string.encode('utf-8', errors='surrogatepass')).hexdigest()

def get_cached_response(cache_key: str) -> Optional[str]:
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data["timestamp"])
        ttl = timedelta(hours=settings.cache_ttl_hours)
        
        if datetime.now() - cached_time > ttl:
            os.remove(cache_file)
            return None
        
        return cache_data["response"]
    except Exception:
        return None

def cache_response(cache_key: str, response: str):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "response": response
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def clear_cache():
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
