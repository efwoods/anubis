import logging

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.error("This is an ERROR message")

