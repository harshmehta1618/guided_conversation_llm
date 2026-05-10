import json
from pathlib import Path

from persona import UserHealthProfile

empty_dict = {}
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'task.json', 'w') as f:
    json.dump(empty_dict, f, indent=4)

with open(BASE_DIR / 'persona.json', 'w') as f:
    json.dump(UserHealthProfile().model_dump(), f, indent=4)

with open(BASE_DIR / 'conversation.json', 'r') as f:
    d = json.load(f)

d['Conversation'] = empty_dict

with open(BASE_DIR / 'conversation.json', 'w') as f:
    json.dump(d, f, indent=4)
