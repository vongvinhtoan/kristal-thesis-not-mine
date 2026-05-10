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


price_df = pl.read_parquet(PROCESSED_DATA_DIR / "price.parquet")
icb_df = pl.read_parquet(PROCESSED_DATA_DIR / "icb.parquet")


crash_df = (
    price_df.join(
        icb_df,
        on="ticker",
        how="left",
        validate="m:1",
    )
    .filter(pl.col("icb2").is_not_null())
    .filter(
        ~pl.col("icb2").is_in(
            [
                "Dịch vụ tài chính",
                "Ngân hàng",
                "Bảo hiểm",
            ]
        )
    )
    .filter(~pl.col("ticker").is_in(["DSC", "MIC"]))
    .with_columns(
        pl.when(pl.col("mktcap") != 0)
        .then(pl.col("mktcap"))
        .otherwise(None)
        .alias("mktcap"),
        year=pl.col("date").dt.year(),
        week=pl.col("date").dt.week(),
        dayofweek=pl.col("date").dt.weekday(),
    )
    .filter(~pl.col("dayofweek").is_in([6, 7]))
    .filter(pl.col("year") >= 2010)
    .filter(pl.col("year") <= 2024)
    .with_columns(pl.col("date").cast(pl.Date))
    .sort(["ticker", "date"])
    .upsample(time_column="date", every="1d", group_by="ticker", maintain_order=True)
    .with_columns(
        dret=(pl.col("close") / pl.col("close").shift(1)).log().over("ticker")
    )
    .filter(pl.col("close").is_not_null() | pl.col("dret").is_not_null())
    .with_columns(
        year=pl.col("date").dt.year(),
        week=pl.col("date").dt.week(),
        mktcap=pl.when(pl.col("mktcap") != 0).then(pl.col("mktcap")).otherwise(None),
    )
    .group_by(["ticker", "year", "week"])
    .agg(
        [
            pl.col("dret").sum(),
            pl.col("mktcap").last(ignore_nulls=True),
            pl.col("icb1").last(ignore_nulls=True),
            pl.col("icb2").last(ignore_nulls=True),
            pl.col("icb3").last(ignore_nulls=True),
            pl.col("icb4").last(ignore_nulls=True),
            pl.col("icb5").last(ignore_nulls=True),
        ]
    )
    # 7. Final Weekly Return and Outlier Filter
    .with_columns(wret=(pl.col("dret").exp() - 1))
    .filter(pl.col("wret").abs() < (pow(1.1, 5) - 1))
    .sort(["ticker", "year", "week"])
    .with_columns(
        weekid=pl.int_range(0, pl.len()).over("ticker"),
        totalmktcap=pl.col("mktcap").sum().over(["year", "week"]),
    )
    .with_columns(mktcap_weighted=pl.col("mktcap") / pl.col("totalmktcap"))
    .pipe(
        lambda x: x.join(
            x.select(
                [
                    pl.col("ticker"),
                    (pl.col("weekid") + 1).alias("weekid"),
                    pl.col("mktcap_weighted").alias("mktcap_weighted_L1"),
                ]
            ),
            on=["ticker", "weekid"],
            how="left",
        )
    )
    .with_columns(wret_weighted=pl.col("mktcap_weighted_L1") * pl.col("wret"))
    .with_columns(vwmktret=pl.col("wret_weighted").sum().over(["year", "week"]))
    .with_columns(
        total_mktcap_ind=pl.col("mktcap").sum().over(["icb1", "year", "week"])
    )
    .with_columns(ind_weight=pl.col("mktcap") / pl.col("total_mktcap_ind"))
    .pipe(
        lambda x: x.join(
            x.select(
                [
                    pl.col("ticker"),
                    (pl.col("weekid") + 1).alias("weekid"),
                    pl.col("ind_weight").alias("ind_weight_L1"),
                ]
            ),
            on=["ticker", "weekid"],
            how="left",
        )
    )
    .with_columns(ind_wret_weighted=pl.col("ind_weight_L1") * pl.col("wret"))
    .with_columns(
        vwindret=pl.col("ind_wret_weighted").sum().over(["icb1", "year", "week"])
    )
    .sort(["ticker", "year", "week"])  # <--- Optional but recommended safety net
    .with_columns(
        vwmktret_L1=pl.col("vwmktret").shift(1).over("ticker"),
        vwmktret_F1=pl.col("vwmktret").shift(-1).over("ticker"),
        vwindret_L1=pl.col("vwindret").shift(1).over("ticker"),
        vwindret_F1=pl.col("vwindret").shift(-1).over("ticker"),
    )
    .with_columns(
        ols_res=pl.col("wret")
        .least_squares.rolling_ols(
            pl.col("vwmktret_L1"),
            pl.col("vwmktret"),
            pl.col("vwmktret_F1"),
            pl.col("vwindret_L1"),
            pl.col("vwindret"),
            pl.col("vwindret_F1"),
            window_size=53,
            min_periods=26,
            add_intercept=True,
            mode="coefficients",
        )
        .over("ticker")
    )
    .filter(pl.col("ols_res").is_not_null())
    .with_columns(
        b_vwmktret_L1=pl.col("ols_res").struct.field("vwmktret_L1"),
        b_vwmktret=pl.col("ols_res").struct.field("vwmktret"),
        b_vwmktret_F1=pl.col("ols_res").struct.field("vwmktret_F1"),
        b_vwindret_L1=pl.col("ols_res").struct.field("vwindret_L1"),
        b_vwindret=pl.col("ols_res").struct.field("vwindret"),
        b_vwindret_F1=pl.col("ols_res").struct.field("vwindret_F1"),
        b_cons=pl.col("ols_res").struct.field("const"),
    )
    .with_columns(
        e=pl.col("wret")
        - (
            pl.col("b_vwmktret_L1") * pl.col("vwmktret_L1")
            + pl.col("b_vwmktret") * pl.col("vwmktret")
            + pl.col("b_vwmktret_F1") * pl.col("vwmktret_F1")
            + pl.col("b_vwindret_L1") * pl.col("vwindret_L1")
            + pl.col("b_vwindret") * pl.col("vwindret")
            + pl.col("b_vwindret_F1") * pl.col("vwindret_F1")
            + pl.col("b_cons")
        )
    )
    .with_columns(W=pl.col("e").log1p())
    .drop_nulls(subset=["W"])
    .drop(["e", "ols_res"])
    .sort(["ticker", "year", "week"])
    .with_columns(
        RET=pl.col("W").mean().over(["ticker", "year"]),
        STDEV=pl.col("W").std().over(["ticker", "year"]),
    )
    .sort(["ticker", "year", "week"])
    .with_columns(
        W2=pl.col("W").pow(2),
        W3=pl.col("W").pow(3),
        W_up2=pl.when(pl.col("W") > pl.col("RET"))
        .then(pl.col("W").pow(2))
        .otherwise(None),
        W_down2=pl.when(pl.col("W") < pl.col("RET"))
        .then(pl.col("W").pow(2))
        .otherwise(None),
        DCRASH=pl.when(pl.col("W").abs() >= 3.09 * pl.col("STDEV"))
        .then(1)
        .otherwise(0),
    )
    .group_by(["ticker", "year"], maintain_order=True)
    .agg(
        [
            pl.col("W").count().alias("W"),
            pl.col("W_up2").count().alias("n_up"),
            pl.col("W_down2").count().alias("n_down"),
            pl.col("W2").sum(),
            pl.col("W3").sum(),
            pl.col("W_up2").sum(),
            pl.col("W_down2").sum(),
            pl.col("DCRASH").sum(),
            pl.col("mktcap").last(ignore_nulls=True),
            pl.col("icb1").last(ignore_nulls=True),
            pl.col("icb2").last(ignore_nulls=True),
            pl.col("icb3").last(ignore_nulls=True),
            pl.col("icb4").last(ignore_nulls=True),
            pl.col("icb5").last(ignore_nulls=True),
            pl.col("RET").last(ignore_nulls=True),
            pl.col("STDEV").last(ignore_nulls=True),
        ]
    )
    .with_columns(
        NCSKEW1=pl.col("W") * (pl.col("W") - 1).pow(1.5) * pl.col("W3"),
        NCSKEW2=(pl.col("W") - 1) * (pl.col("W") - 2) * pl.col("W2").pow(1.5),
    )
    .with_columns(
        NCSKEW=-pl.col("NCSKEW1") / pl.col("NCSKEW2"),
        DUVOL=(
            ((pl.col("n_up") - 1) * pl.col("W_down2"))
            / ((pl.col("n_down") - 1) * pl.col("W_up2"))
        ).log(),
        DCRASH=pl.when(pl.col("DCRASH") > 0).then(1).otherwise(0),
    )
    .drop(["W2", "W3", "W_up2", "W_down2", "NCSKEW1", "NCSKEW2", "n_up", "n_down"])
)

print(crash_df.schema)
print(crash_df.describe().to_dicts())

crash_df.write_parquet(PROCESSED_DATA_DIR / "crash.parquet")
