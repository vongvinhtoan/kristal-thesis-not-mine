from pathlib import Path

import pandas as pd
import polars as pl
import statsmodels.api as sm
from _logging import get_logger
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
from scipy.stats import pearsonr

logger = get_logger("preprocessing")

SCRIPT_DIR = Path(__file__).parent.absolute().resolve()
DATA_DIR = (SCRIPT_DIR.parent / "data").absolute().resolve()
PROCESSED_DATA_DIR = (SCRIPT_DIR.parent / "processed").absolute().resolve()

# 1. Load the dataset
controls_df = pl.read_parquet(PROCESSED_DATA_DIR / "obs_data.parquet")

dep_vars = ["NCSKEW", "DUVOL"]

# 2. Define your variable groups cleanly
base_controls = [
    "LNCSKEW",
    "LSIZE",
    "LMTB",
    "LLEVRG",
    "LROA",
    "LABACC",
    "LRET",
    "LSTDEV",
    "LDTURN",
]
all_gpr_vars = ["GPR", "GPRA", "GPRT"]

# 3. Clean the data (Drop nulls across ALL variables so the sample size matches exactly)
keep_cols = dep_vars + base_controls + all_gpr_vars + ["ticker", "year", "icb2"]
reg_df_pl = controls_df.select(keep_cols).drop_nulls(subset=keep_cols)

n_tickers = reg_df_pl.select(pl.col("ticker").n_unique()).item()
print(f"Number of distinct tickers: {n_tickers}")

print(f"\n{'#' * 70}")
print("### Observations")
print(f"{'#' * 70}")
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(-1)
print(reg_df_pl.describe())

# 4. Convert to Pandas and set the Panel Index
reg_df = reg_df_pl.to_pandas()
reg_df = reg_df.set_index(["ticker", "year"])

# Print Correlation Matrix
# 1. Define the variables in the exact order you want them in the matrix
# Mapping the dataset names to the presentation names from the image
correlation_vars = [
    "NCSKEW",
    "DUVOL",
    "LNCSKEW",
    "LSIZE",
    "LMTB",
    "LLEVRG",
    "LROA",
    "LABACC",
    "LRET",
    "LSTDEV",
    "LDTURN",
]

display_names = [
    "(1)NCSKEW_t",
    "(2)DUVOL_t",
    "(3)RM_t-1",
    "(4)SIZE_t-1",
    "(5)MTB_t-1",
    "(6)LEVRG_t-1",
    "(7)ROA_t-1",
    "(8)ABACC_t-1",
    "(9)RET_t-1",
    "(10)STDEV_t-1",
    "(11)DTURN_t-1",
]

# Extract relevant data
df_corr = reg_df[correlation_vars].copy()
n_vars = len(correlation_vars)

# 2. Create an empty DataFrame to store the formatted strings
matrix_df = pd.DataFrame(
    "", index=display_names, columns=[f"({i + 1})" for i in range(n_vars)]
)

# 3. Calculate pairwise correlations and p-values
for i in range(n_vars):
    for j in range(i + 1):  # i+1 ensures we include the diagonal and lower triangle
        v1 = correlation_vars[i]
        v2 = correlation_vars[j]

        if i == j:
            # Diagonal elements
            matrix_df.iloc[i, j] = "1.000"
        else:
            # Calculate correlation coefficient and p-value
            corr, p_val = pearsonr(df_corr[v1], df_corr[v2])

            # Determine significance stars
            if p_val < 0.01:
                stars = "***"
            elif p_val < 0.05:
                stars = "**"
            elif p_val < 0.1:
                stars = "*"
            else:
                stars = ""

            # Format outputs to 3 decimal places
            corr_str = f"{corr:.3f}{stars}"
            pval_str = f"({p_val:.3f})"

            # Combine them with a newline block for formatting
            matrix_df.iloc[i, j] = f"{corr_str}\n{pval_str}"

# 4. Print or save the output
print(f"\n{'#' * 70}\n### CORRELATION MATRIX\n{'#' * 70}")
# Configure pandas to display multiline strings nicely inside columns
pd.set_option("display.max_colwidth", None)
print(matrix_df.to_string())

# 5. Define the exact variable lists for the two separate regressions
indep_vars_aggregate = ["GPR"] + base_controls
indep_vars_split = ["GPRA", "GPRT"] + base_controls

# Add intercepts for both models
X_agg = sm.add_constant(reg_df[indep_vars_aggregate])
X_split = sm.add_constant(reg_df[indep_vars_split])

# 6. Run the Regressions
for y_var in dep_vars:
    print(f"\n{'#' * 70}")
    print(f"### DEPENDENT VARIABLE: {y_var}")
    print(f"{'#' * 70}")

    Y = reg_df[y_var]

    # =========================================================
    # SET A: AGGREGATE GPR EFFECT
    # =========================================================
    print(f"\n{'-' * 50}\n PART A: OVERALL GPR EFFECT\n{'-' * 50}")

    # Model A1: Industry FE
    mod_agg_ind = PanelOLS(Y, X_agg, time_effects=False, other_effects=reg_df[["icb2"]])
    res_agg_ind = mod_agg_ind.fit(cov_type="clustered", cluster_entity=True)
    print("\n--- Model A1: Industry FE (Aggregate) ---")
    print(res_agg_ind.summary)

    # Model A2: Firm FE
    mod_agg_firm = PanelOLS(Y, X_agg, entity_effects=True, time_effects=False)
    res_agg_firm = mod_agg_firm.fit(cov_type="clustered", cluster_entity=True)
    print("\n--- Model A2: Firm FE (Aggregate) ---")
    print(res_agg_firm.summary)

    # =========================================================
    # SET B: SPLIT EFFECT (ACTS vs. THREATS)
    # =========================================================
    print(f"\n{'-' * 50}\n PART B: SPLIT EFFECT (GPRA & GPRT)\n{'-' * 50}")

    # Model B1: Industry FE
    mod_split_ind = PanelOLS(
        Y, X_split, time_effects=False, other_effects=reg_df[["icb2"]]
    )
    res_split_ind = mod_split_ind.fit(cov_type="clustered", cluster_entity=True)
    print("\n--- Model B1: Industry FE (Split) ---")
    print(res_split_ind.summary)

    # Model B2: Firm FE
    mod_split_firm = PanelOLS(Y, X_split, entity_effects=True, time_effects=False)
    res_split_firm = mod_split_firm.fit(cov_type="clustered", cluster_entity=True)
    print("\n--- Model B2: Firm FE (Split) ---")
    print(res_split_firm.summary)
