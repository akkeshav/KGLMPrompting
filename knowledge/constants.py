from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('config.yml')
OPENAI_API_KEY = ''

NUMERSENSE_ANSWERS = ['no', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
DATASET_TO_QUERY_KEY = {
    'numersense': 'query',
    'CSQA': 'query',
    'CSQA2': 'query',
    'qasc': 'query',
}
