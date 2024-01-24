"""Save all adsorption energy results into a db."""
import json
import pickle

from pathlib import Path
from tqdm import tqdm

import redis

redis_db = redis.Redis(host='localhost', port=6379, db=1)

for p in tqdm(Path("/dev/shm/chemreasoner/catalysis").rglob("*/adsorption.json")):
    
    with open(str(p), "r") as f:
        data = json.load(f)

    redis_db.set(str(p), json.dumps(data))

for p in tqdm(Path("/dev/shm/chemreasoner/catalysis", "slabs").rglob("*.pkl")):
    
    with open(str(p), "rb") as f:
        data = pickle.load(f)

    redis_db.set(str(p), pickle.dumps(data))