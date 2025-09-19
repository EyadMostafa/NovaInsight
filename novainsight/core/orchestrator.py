from novainsight.utils.logger import get_logger

logger = get_logger(__name__)

print(logger.level)
logger.warning("test")