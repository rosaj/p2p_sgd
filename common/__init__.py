import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from common.util import *
from common.model import *
import environ
if environ.is_collab():
    pass
else:
    pass
