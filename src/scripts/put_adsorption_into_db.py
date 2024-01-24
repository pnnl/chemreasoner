"""Save all adsorption energy results into a db."""
import json

from pathlib import Path
from tqdm import tqdm

import redis

redis_db = redis.Redis(host='localhost', port=6379, db=0)

for p in tqdm(Path("/qfs/projects/aqe_tec4/catalysis/chemreasoner/").rglob("*/adsorption.json")):
    
    with open(str(p), "r") as f:
        data = json.load(f)

    redis_db.set(str(p), json.dumps(data))