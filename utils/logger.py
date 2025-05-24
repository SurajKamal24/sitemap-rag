import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.DEBUG, 
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler() 
    ]
)

def get_logger(name):
    return logging.getLogger(name)

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
