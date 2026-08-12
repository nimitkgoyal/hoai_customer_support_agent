import logging
import os

# Define absolute paths relative to our project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure logs directory exists safely
os.makedirs(LOG_DIR, exist_ok=True)

# Define a professional, scannable logging format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"

# Setup root logger configurations
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  # Prints out cleanly to your live console window
    ]
)

# Export our configured logger instance
logger = logging.getLogger("hoai_agent")
