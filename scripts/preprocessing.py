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


CLEANUP_PRICE_DIR = CLEANUP_DIR / "price"
CLEANUP_PRICE_DIR.mkdir(exist_ok=True, parents=True)
for file in (DATA_DIR / "HOSE/Excel").iterdir():
    cleaned_up_path = CLEANUP_PRICE_DIR / f"{file.stem}.parquet"

    if not file.is_file() or file.suffix != ".xlsx":
        continue
    if cleaned_up_path.exists():
        logger.info(f"[PRICE] Skipping {file.name}, already processed")
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

    logger.info(f"[PRICE] Saving cleaned up data to {cleaned_up_path}")

    df.write_parquet(cleaned_up_path)


price_df_path = PROCESSED_DATA_DIR / "price.parquet"
if not price_df_path.exists():
    price_df = pl.concat(
        [pl.read_parquet(f) for f in CLEANUP_PRICE_DIR.glob("*.parquet")]
    )
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


CLEANUP_FINANCIALS_DIR = CLEANUP_DIR / "financials"
CLEANUP_FINANCIALS_DIR.mkdir(exist_ok=True, parents=True)
for file in (DATA_DIR / "BCTC/XLXS").iterdir():
    cleaned_up_path = CLEANUP_FINANCIALS_DIR / f"{file.stem}.parquet"

    if not file.is_file() or file.suffix != ".xlsx":
        continue
    if cleaned_up_path.exists():
        logger.info(f"[FINANCIALS] Skipping {file.name}, already processed")
        continue

    rename_mapping = {
        "Mã": "ticker",
        "Sàn": "exchange",
        "I. TÀI SẢN NGẮN HẠN\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "CA",
        "1. Tiền và tương đương tiền \nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "cash",
        "3. Các khoản phải thu ngắn hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "s_rec",
        "3.1. Phải thu ngắn hạn của khách hàng\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "s_traderec",
        "4. Hàng tồn kho, ròng\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "inventory",
        "II. TÀI SẢN DÀI HẠN\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "NCA",
        "1. Phải thu dài hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "l_rec",
        "1.1. Phải thu khách hang dài hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "l_traderec",
        "1.1. Phải thu khách hàng dài hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "l_traderec",
        "2.1. GTCL TSCĐ hữu hình\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "ppe",
        "A. TỔNG CỘNG TÀI SẢN\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "TA",
        "I. NỢ PHẢI TRẢ\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "TL",
        "1. Nợ ngắn hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "CL",
        "1.1. Phải trả người bán ngắn hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "s_tradepay",
        "2. Nợ dài hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "NCL",
        "2.1. Phải trả nhà cung cấp dài hạn\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "l_tradepay",
        "II. VỐN CHỦ SỞ HỮU\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "equity",
        "1.12. Lãi chưa phân phối\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "RE",
        "1. Doanh thu bán hàng và cung cấp dịch vụ\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "sales",
        "3. Doanh thu thuần\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "netsales",
        "4. Giá vốn hàng bán\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "cogs",
        "5. Lợi nhuận gộp về bán hàng và cung cấp dịch vụ\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "grossprofit",
        "9. Chi phí bán hàng\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "sellingexp",
        "10. Chi phí quản lý doanh  nghiệp\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "GAexp",
        "10. Chi phí quản lý doanh nghiệp\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "GAexp",
        "11. Lợi nhuận thuần từ hoạt động kinh doanh\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "operatingprofit",
        "16. Tổng lợi nhuận kế toán trước thuế\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "ebt",
        "18. Lợi nhuận sau thuế thu nhập doanh nghiệp\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "netprofit",
        "I. Lưu chuyển tiền thuần từ hoạt động kinh doanh (GT)\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "cfo",
        "2. Khấu hao TSCĐ và BĐSĐT (GT)\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "dep",
        "6. Cổ tức đã trả (GT)\nHợp nhất\nQuý: Hàng năm\n\nĐơn vị: VND": "div",
    }

    df = (
        pl.read_excel(file)
        .rename(rename_mapping, strict=False)
        .select(list(set(rename_mapping.values())))
    )

    logger.info(f"[FINANCIALS] Saving cleaned up data to {cleaned_up_path}")

    df.write_parquet(cleaned_up_path)

financials_df_path = PROCESSED_DATA_DIR / "financials.parquet"
if not financials_df_path.exists():
    financials_df = pl.concat(
        [pl.read_parquet(f) for f in CLEANUP_FINANCIALS_DIR.glob("*.parquet")]
    ).filter(
        pl.col("ticker").is_not_null() & pl.col("exchange").is_in(["HNX", "HOSE"])
    )
    financials_df.write_parquet(financials_df_path)
    logger.info(f"Saved {financials_df_path.name} ({financials_df.height} rows)")
else:
    logger.info(f"Skipping {financials_df_path.name}, already exists")


gpr_index_df_path = PROCESSED_DATA_DIR / "gpr_index.parquet"
if not gpr_index_df_path.exists():
    gpr_index_df = pl.read_excel(
        DATA_DIR / "data_gpr_export.xls", columns=["month", "GPR", "GPRT", "GPRA"]
    ).with_columns([
        pl.col("month"),
        pl.col("GPR").cast(pl.Float64),
        pl.col("GPRA").cast(pl.Float64),
        pl.col("GPRT").cast(pl.Float64),
    ])
    gpr_index_df.write_parquet(gpr_index_df_path)
else:
    logger.info(f"Skipping {gpr_index_df_path.name}, already exists")
