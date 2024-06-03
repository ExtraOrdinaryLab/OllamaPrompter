import tqdm
import logging
from rich.logging import RichHandler
from colorlog import ColoredFormatter


LOG_LEVEL = logging.NOTSET
LOGFORMAT = "%(log_color)s%(asctime)s%(reset)s - %(log_color)s%(levelname)s%(reset)s - %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
# logger.addHandler(stream)
logger.addHandler(RichHandler())