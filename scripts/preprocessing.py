import polars as pl
from pathlib import Path

from _logging import get_logger

logger = get_logger("preprocessing")

SCRIPT_DIR = Path(__file__).parent.absolute().resolve()

DATA_DIR = (SCRIPT_DIR.parent / "data").absolute().resolve()
PROCESSED_DATA_DIR = (SCRIPT_DIR.parent / "processed").absolute().resolve()

logger.info(f"{DATA_DIR=}")
logger.info(f"{PROCESSED_DATA_DIR=}")
assert DATA_DIR.is_dir()
assert PROCESSED_DATA_DIR.is_dir()


CLEANUP_DIR = PROCESSED_DATA_DIR / "cleanup"
CLEANUP_DIR.mkdir(parents=True, exist_ok=True)


for file in (DATA_DIR / "HOSE/Excel").iterdir():
    cleaned_up_path = CLEANUP_DIR / f"{file.stem}.parquet"

    if not file.is_file() or file.suffix != ".xlsx":
        continue
    if cleaned_up_path.exists():
        logger.info(f"Skipping {file.name}, already processed")
        continue

    rename_mapping = {
        "Ticker": "ticker",
        "Date": "date",
        "Close Adjusted (D)\nUnit: VND": "close",
        "Total trading volume (D)\nUnit: Shares": "volume",
        "Current Outstanding Shares\nUnit: Shares": "shares",
        "Market Capitalization \nUnit: VND": "mktcap",
    }

    dtype_mapping = {
        "ticker": pl.String,
        "date": pl.Date,
        "close": pl.Float64,
        "volume": pl.Int64,
        "shares": pl.Int64,
        "mktcap": pl.Int64,
    }

    wanted_columns = list(rename_mapping.keys())

    df = (
        pl.read_excel(
            file,
            columns=list(rename_mapping.keys()),
        )
        .rename(rename_mapping)
        .with_columns(
            [pl.col(name).cast(dtype) for name, dtype in dtype_mapping.items()]
        )
    )

    cleaned_up_path = CLEANUP_DIR / f"{file.stem}.parquet"

    logger.info(f"Saving cleaned up data to {cleaned_up_path}")

    df.write_parquet(cleaned_up_path)


price_df_path = PROCESSED_DATA_DIR / "price.parquet"
if not price_df_path.exists():
    price_df = pl.concat([pl.read_parquet(f) for f in CLEANUP_DIR.glob("*.parquet")])
    price_df.write_parquet(PROCESSED_DATA_DIR / "price.parquet")
else:
    logger.info(f"Skipping {price_df_path.name}, already exists")


icb_df_path = PROCESSED_DATA_DIR / "icb.parquet"
if not icb_df_path.exists():
    rename_mapping = {
        "Mã": "ticker",
        "Phân ngành - ICB L1": "icb1",
        "Phân ngành - ICB L2": "icb2",
        "Phân ngành - ICB L3": "icb3",
        "Phân ngành - ICB L4": "icb4",
        "Phân ngành - ICB L5": "icb5",
    }

    icb_df = (
        pl.read_excel(DATA_DIR / "INFO/ICB.xlsx", columns=list(rename_mapping.keys()))
        .rename(rename_mapping)
        .filter(pl.col("ticker") != "")
    )
    icb_df.write_parquet(icb_df_path)
else:
    logger.info(f"Skipping {icb_df_path.name}, already exists")

