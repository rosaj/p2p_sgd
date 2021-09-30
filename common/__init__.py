from common.util import *
from common.model import *
import environ
if environ.is_collab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
