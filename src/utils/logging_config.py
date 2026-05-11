import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,  # Enable DEBUG for detailed info
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('recsys.log', encoding='utf-8')
        ]
    )
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    return logging.getLogger(__name__)

logger = setup_logging()