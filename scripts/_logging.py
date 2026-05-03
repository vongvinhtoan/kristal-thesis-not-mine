import logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute().resolve()

LOG_DIR = (SCRIPT_DIR.parent / "logs").absolute().resolve()

def get_iso_timestampt(utc: int = 7) -> str:
    from datetime import datetime, timezone, timedelta

    tz = timezone(timedelta(hours=utc))
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(
            LOG_DIR / f"{name}_{get_iso_timestampt()}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)
        stream_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
