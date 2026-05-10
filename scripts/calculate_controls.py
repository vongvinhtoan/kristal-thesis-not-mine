import polars as pl
import polars_ols
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


financials_df = pl.read_parquet(PROCESSED_DATA_DIR / "financials.parquet")
crash_df = pl.read_parquet(PROCESSED_DATA_DIR / "crash.parquet")
turnover_df = pl.read_parquet(PROCESSED_DATA_DIR / "turnover.parquet")

controls_df = (
    financials_df.join(crash_df, on=["ticker", "year"], how="left")
    .join(turnover_df, on=["ticker", "year"], how="left")
    .sort(["ticker", "year"])
    .with_columns(
        SIZE=pl.col("TA").log(),
        MTB=pl.col("mktcap") / pl.col("equity"),
        LEVRG=pl.col("TL") / pl.col("TA"),
        ROA=pl.col("netprofit") / pl.col("TA"),
        TA_L1=pl.col("TA").shift(1).over("ticker"),
        D_sales=pl.col("sales") - pl.col("sales").shift(1).over("ticker"),
        D_s_rec=pl.col("s_rec") - pl.col("s_rec").shift(1).over("ticker"),
        DTURN=pl.col("turnover") - pl.col("turnover").shift(1).over("ticker"),
    )
    .with_columns(
        inv_TA=1 / pl.col("TA_L1"),
        accruals=pl.col("netprofit") - pl.col("cfo"),
    )
    .with_columns(
        accruals_scaled=pl.col("accruals") * pl.col("inv_TA"),
        Drev_scaled=pl.col("D_sales") * pl.col("inv_TA"),
        Drec_scaled=pl.col("D_s_rec") * pl.col("inv_TA"),
        ppe_scaled=pl.col("ppe") * pl.col("inv_TA"),
    )
    .with_columns(
        ols_coefs=pl.col("accruals_scaled")
        .least_squares.ols(
            pl.col("inv_TA"),
            pl.col("Drev_scaled"),
            pl.col("ppe_scaled"),
            add_intercept=True,
            mode="coefficients",
        )
        .over(["icb1", "year"])
    )
    .with_columns(
        b_cons=pl.col("ols_coefs").struct.field("const"),
        b_inv_TA=pl.col("ols_coefs").struct.field("inv_TA"),
        b_Drev_scaled=pl.col("ols_coefs").struct.field("Drev_scaled"),
        b_ppe_scaled=pl.col("ols_coefs").struct.field("ppe_scaled"),
    )
    .with_columns(
        NACC=(
            pl.col("b_cons")
            + (pl.col("b_inv_TA") * pl.col("inv_TA"))
            + (
                pl.col("b_Drev_scaled")
                * (pl.col("Drev_scaled") - pl.col("Drec_scaled"))
            )
            + (pl.col("b_ppe_scaled") * pl.col("ppe_scaled"))
        )
    )
    .with_columns(DISACC=pl.col("accruals_scaled") - pl.col("NACC"))
    .with_columns(ABACC=pl.col("DISACC").abs())
    .sort(["ticker", "year"])
    .with_columns(
        LSIZE=pl.col("SIZE").shift(1).over("ticker"),
        LMTB=pl.col("MTB").shift(1).over("ticker"),
        LLEVRG=pl.col("LEVRG").shift(1).over("ticker"),
        LROA=pl.col("ROA").shift(1).over("ticker"),
        LABACC=pl.col("ABACC").shift(1).over("ticker"),
        LRET=pl.col("RET").shift(1).over("ticker"),
        LSTDEV=pl.col("STDEV").shift(1).over("ticker"),
        LDTURN=pl.col("DTURN").shift(1).over("ticker"),
        LNCSKEW=pl.col("NCSKEW").shift(1).over("ticker"),
    )
    .drop(
        [
            "TA_L1",
            "D_sales",
            "D_s_rec",
            "inv_TA",
            "accruals",
            "accruals_scaled",
            "Drev_scaled",
            "Drec_scaled",
            "ppe_scaled",
            "ols_coefs",
            "b_cons",
            "b_inv_TA",
            "b_Drev_scaled",
            "b_ppe_scaled",
            "NACC",
            "DISACC",
        ]
    )
)

print("\n".join(controls_df.columns))
controls_df.write_parquet(PROCESSED_DATA_DIR / "controls.parquet")
