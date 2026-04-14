import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


st.set_page_config(page_title="Paired Tests / Repeated Measures ANOVA", layout="wide")
st.title("Paired Tests / Repeated Measures ANOVA")
st.caption(
    "2 conditions: Shapiro-Wilk on paired differences / paired t-test / Wilcoxon signed-rank, "
    "3+ conditions: repeated measures ANOVA / post hoc tests"
)
st.info(
    "This app assumes a wide-format CSV: each row is one subject and each column is one condition."
)

ALPHA_OPTIONS = [0.01, 0.05, 0.10]

SAMPLE_TWO_CSV = """before,after
12.3,11.2
11.8,10.4
13.1,11.9
12.2,11.5
11.9,10.9
12.7,11.3
"""

SAMPLE_THREE_CSV = """T0,T1,T2
12.3,11.8,11.4
11.8,11.1,10.6
13.1,12.5,12.0
12.2,11.7,11.0
11.9,11.3,10.7
12.7,12.0,11.4
"""


def load_csv_flex(uploaded_file):
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception:
            continue
    raise ValueError("Failed to read CSV with supported encodings: utf-8, utf-8-sig, cp932")


def to_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def summarize(series, name):
    x = pd.Series(series).dropna()
    if x.empty:
        return {
            "Condition": name,
            "N": 0,
            "Mean": np.nan,
            "SD": np.nan,
            "Min": np.nan,
            "Q1": np.nan,
            "Median": np.nan,
            "Q3": np.nan,
            "Max": np.nan,
        }

    return {
        "Condition": name,
        "N": int(x.shape[0]),
        "Mean": float(x.mean()),
        "SD": float(x.std(ddof=1)) if len(x) >= 2 else np.nan,
        "Min": float(x.min()),
        "Q1": float(x.quantile(0.25)),
        "Median": float(x.median()),
        "Q3": float(x.quantile(0.75)),
        "Max": float(x.max()),
    }


def safe_shapiro(x):
    x = pd.Series(x).dropna()

    if len(x) < 3:
        return np.nan, "Not run: n < 3"
    if len(x) > 5000:
        return np.nan, "Not run: n > 5000"
    if x.nunique() < 2:
        return np.nan, "Not run: differences are nearly constant"

    try:
        result = stats.shapiro(x)
        return float(result.pvalue), ""
    except Exception as e:
        return np.nan, f"Failed: {e}"


def interpret_normality(p, alpha):
    if pd.isna(p):
        return "Not assessed"
    return "Non-normality suggested" if p < alpha else "No strong evidence against normality"


def interpret_significance(p, alpha):
    if pd.isna(p):
        return "Not assessed"
    return "Significant" if p < alpha else "Not significant"


def append_result(rows, section, test, p_value, alpha, interpretation, note="", recommended=False):
    rows.append(
        {
            "Section": section,
            "Test": test,
            "PValue": p_value,
            "Alpha": alpha,
            "Recommended": "Yes" if recommended else "",
            "Interpretation": interpretation,
            "Note": note if note else "None",
        }
    )


def run_paired_tests(df_two, alpha=0.05):
    rows = []
    col1, col2 = df_two.columns.tolist()
    clean = df_two[[col1, col2]].dropna().copy()

    x = clean[col1]
    y = clean[col2]
    diff = x - y

    shapiro_p, shapiro_note = safe_shapiro(diff)
    append_result(
        rows=rows,
        section="Assumption Check",
        test="Shapiro-Wilk (paired differences)",
        p_value=shapiro_p,
        alpha=alpha,
        interpretation=interpret_normality(shapiro_p, alpha),
        note=shapiro_note,
    )

    primary_test = "Paired t-test" if pd.notna(shapiro_p) and shapiro_p >= alpha else "Wilcoxon signed-rank"

    if len(clean) >= 2:
        try:
            t_res = stats.ttest_rel(x, y, alternative="two-sided", nan_policy="omit")
            append_result(
                rows=rows,
                section="Comparison",
                test="Paired t-test",
                p_value=float(t_res.pvalue),
                alpha=alpha,
                interpretation=interpret_significance(float(t_res.pvalue), alpha),
                recommended=(primary_test == "Paired t-test"),
            )
        except Exception as e:
            append_result(
                rows=rows,
                section="Comparison",
                test="Paired t-test",
                p_value=np.nan,
                alpha=alpha,
                interpretation="Not assessed",
                note=f"Failed: {e}",
                recommended=(primary_test == "Paired t-test"),
            )
    else:
        append_result(
            rows=rows,
            section="Comparison",
            test="Paired t-test",
            p_value=np.nan,
            alpha=alpha,
            interpretation="Not assessed",
            note="At least 2 complete pairs are required",
            recommended=(primary_test == "Paired t-test"),
        )

    if len(clean) >= 1:
        try:
            w_res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            append_result(
                rows=rows,
                section="Comparison",
                test="Wilcoxon signed-rank",
                p_value=float(w_res.pvalue),
                alpha=alpha,
                interpretation=interpret_significance(float(w_res.pvalue), alpha),
                recommended=(primary_test == "Wilcoxon signed-rank"),
            )
        except Exception as e:
            append_result(
                rows=rows,
                section="Comparison",
                test="Wilcoxon signed-rank",
                p_value=np.nan,
                alpha=alpha,
                interpretation="Not assessed",
                note=f"Failed: {e}",
                recommended=(primary_test == "Wilcoxon signed-rank"),
            )
    else:
        append_result(
            rows=rows,
            section="Comparison",
            test="Wilcoxon signed-rank",
            p_value=np.nan,
            alpha=alpha,
            interpretation="Not assessed",
            note="Complete paired data are required",
            recommended=(primary_test == "Wilcoxon signed-rank"),
        )

    return pd.DataFrame(rows), clean, diff, primary_test, shapiro_p


def wide_to_long_complete(df_wide):
    complete = df_wide.dropna().copy().reset_index(drop=True)
    complete.insert(0, "Subject", np.arange(1, len(complete) + 1))
    long_df = complete.melt(id_vars="Subject", var_name="Condition", value_name="Value")
    return complete, long_df


def run_rm_anova(df_selected, alpha=0.05, correction_method="holm"):
    complete, long_df = wide_to_long_complete(df_selected)

    anova_table = None
    anova_note = ""

    try:
        model = AnovaRM(data=long_df, depvar="Value", subject="Subject", within=["Condition"])
        fitted = model.fit()
        anova_table = fitted.anova_table.reset_index().rename(columns={"index": "Effect"})
    except Exception as e:
        anova_note = f"Repeated measures ANOVA failed: {e}"

    pairs = list(itertools.combinations(df_selected.columns.tolist(), 2))
    raw_pvals = []
    interim = []

    for a, b in pairs:
        pair = complete[[a, b]].dropna()
        if len(pair) < 2:
            interim.append((a, b, len(pair), np.nan, np.nan, "At least 2 complete pairs are required"))
            raw_pvals.append(np.nan)
            continue

        try:
            res = stats.ttest_rel(pair[a], pair[b], alternative="two-sided")
            p = float(res.pvalue)
            mean_diff = float(pair[a].mean() - pair[b].mean())
            interim.append((a, b, len(pair), mean_diff, p, ""))
            raw_pvals.append(p)
        except Exception as e:
            interim.append((a, b, len(pair), np.nan, np.nan, f"Failed: {e}"))
            raw_pvals.append(np.nan)

    corrected = [np.nan] * len(raw_pvals)
    reject = [False] * len(raw_pvals)
    valid_indices = [i for i, p in enumerate(raw_pvals) if pd.notna(p)]

    if valid_indices:
        valid_pvals = [raw_pvals[i] for i in valid_indices]
        rej, corr_p, _, _ = multipletests(valid_pvals, alpha=alpha, method=correction_method)
        for idx, p_corr, rj in zip(valid_indices, corr_p, rej):
            corrected[idx] = float(p_corr)
            reject[idx] = bool(rj)

    pairwise_rows = []
    for i, (a, b, n_pair, mean_diff, raw_p, note) in enumerate(interim):
        pairwise_rows.append(
            {
                "Comparison": f"{a} vs {b}",
                "N": n_pair,
                "MeanDiff": mean_diff,
                "RawPValue": raw_p,
                f"AdjustedPValue_{correction_method}": corrected[i],
                "Interpretation": interpret_significance(corrected[i], alpha) if pd.notna(corrected[i]) else "Not assessed",
                "Note": note if note else "None",
            }
        )

    return anova_table, pd.DataFrame(pairwise_rows), complete, anova_note


def plot_box(df_selected):
    fig, ax = plt.subplots(figsize=(max(5, len(df_selected.columns) * 1.2), 3.6))
    values = [df_selected[col].dropna() for col in df_selected.columns]
    ax.boxplot(values, tick_labels=df_selected.columns.tolist())
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spaghetti(complete_df):
    fig, ax = plt.subplots(figsize=(max(5, (complete_df.shape[1] - 1) * 1.2), 3.8))
    conds = complete_df.columns.tolist()[1:]
    x = np.arange(len(conds))

    for _, row in complete_df.iterrows():
        ax.plot(x, row[conds].values, marker="o", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Value")
    ax.set_title("Individual Trajectories")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


st.sidebar.header("Settings")
alpha = st.sidebar.selectbox(
    "Significance level α",
    options=ALPHA_OPTIONS,
    index=1,
    format_func=lambda x: f"{x:.2f}",
)
correction_method = st.sidebar.selectbox(
    "Multiple-comparison correction (post hoc for 3+ conditions)",
    options=["holm", "bonferroni", "fdr_bh"],
    index=0,
)

with st.expander("CSV examples", expanded=True):
    st.markdown(
        "Each row should represent one subject, and each column should represent one condition. "
        "Rows with missing values are excluded from analyses requiring complete paired data."
    )
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**2-condition example**")
        st.code(SAMPLE_TWO_CSV, language="csv")
        st.download_button(
            label="Download sample CSV (2 conditions)",
            data=SAMPLE_TWO_CSV.encode("utf-8-sig"),
            file_name="sample_paired_two_conditions.csv",
            mime="text/csv",
        )

    with c2:
        st.markdown("**3+-condition example**")
        st.code(SAMPLE_THREE_CSV, language="csv")
        st.download_button(
            label="Download sample CSV (3+ conditions)",
            data=SAMPLE_THREE_CSV.encode("utf-8-sig"),
            file_name="sample_repeated_measures.csv",
            mime="text/csv",
        )

uploaded_file = st.file_uploader("Upload a wide-format CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

try:
    df = load_csv_flex(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("At least 2 columns are required.")
    st.stop()

st.subheader("Loaded data")
st.dataframe(df.head(30), use_container_width=True)

selected_cols = st.multiselect(
    "Select condition columns for analysis (2 columns: paired test, 3+ columns: repeated measures ANOVA)",
    options=df.columns.tolist(),
    default=df.columns.tolist()[: min(3, len(df.columns))],
)

if len(selected_cols) < 2:
    st.warning("Please select at least 2 columns.")
    st.stop()

selected_df = df[selected_cols].copy()
for col in selected_cols:
    selected_df[col] = to_numeric_series(selected_df[col])

non_numeric_counts = {}
for col in selected_cols:
    original_non_na = df[col].notna().sum()
    numeric_non_na = selected_df[col].notna().sum()
    non_numeric_counts[col] = int(original_non_na - numeric_non_na)

if any(v > 0 for v in non_numeric_counts.values()):
    msg = ", ".join(f"{k}: {v}" for k, v in non_numeric_counts.items() if v > 0)
    st.warning(f"Non-numeric values were removed during conversion ({msg}).")

st.subheader("Descriptive statistics")
summary_df = pd.DataFrame([summarize(selected_df[col], col) for col in selected_cols])
st.dataframe(summary_df, use_container_width=True)

st.subheader("Boxplot")
fig_box = plot_box(selected_df)
st.pyplot(fig_box)
plt.close(fig_box)

if len(selected_cols) == 2:
    st.subheader("Paired comparison (2 conditions)")
    results_df, clean_df, diff, primary_test, shapiro_p = run_paired_tests(selected_df, alpha=alpha)

    st.write(f"Number of complete pairs: **{len(clean_df)}**")
    st.dataframe(results_df, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Histogram of paired differences**")
        fig_hist, ax_hist = plt.subplots(figsize=(4.5, 3.2))
        ax_hist.hist(diff.dropna(), bins="auto")
        ax_hist.set_xlabel(f"Difference ({selected_cols[0]} - {selected_cols[1]})")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with c2:
        st.markdown("**Q-Q plot of paired differences**")
        fig_qq, ax_qq = plt.subplots(figsize=(4.5, 3.2))
        if diff.dropna().shape[0] >= 2:
            stats.probplot(diff.dropna(), dist="norm", plot=ax_qq)
            ax_qq.grid(True, alpha=0.3)
        else:
            ax_qq.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax_qq.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_qq)
        plt.close(fig_qq)

    if pd.notna(shapiro_p) and shapiro_p >= alpha:
        st.info("Normality of paired differences was not strongly rejected, so the paired t-test is the primary option.")
    else:
        st.info("Normality of paired differences is questionable or inconclusive, so the Wilcoxon signed-rank test is the primary option.")

    st.caption("Rows with a missing value in either condition are excluded from paired analysis.")

    result_csv = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download test results as CSV",
        data=result_csv,
        file_name="paired_test_results.csv",
        mime="text/csv",
    )

else:
    st.subheader("Repeated measures ANOVA (one-factor)")
    anova_table, pairwise_df, complete_df, anova_note = run_rm_anova(
        selected_df,
        alpha=alpha,
        correction_method=correction_method,
    )

    st.write(f"Number of complete subjects: **{len(complete_df)}**")

    fig_spaghetti = plot_spaghetti(complete_df)
    st.pyplot(fig_spaghetti)
    plt.close(fig_spaghetti)

    if anova_note:
        st.error(anova_note)
    else:
        st.markdown("**ANOVA results**")
        st.dataframe(anova_table, use_container_width=True)

    st.markdown("**Post hoc comparisons (paired t-tests with multiplicity correction)**")
    st.dataframe(pairwise_df, use_container_width=True)
    st.info(
        "This implementation performs one-factor repeated measures ANOVA. "
        "Post hoc testing is done using paired t-tests with multiple-comparison correction."
    )
    st.caption("Only subjects with complete data across all selected conditions are included.")

    anova_csv = anova_table.to_csv(index=False).encode("utf-8-sig") if anova_table is not None else b""
    pairwise_csv = pairwise_df.to_csv(index=False).encode("utf-8-sig")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download ANOVA results as CSV",
            data=anova_csv,
            file_name="repeated_measures_anova_results.csv",
            mime="text/csv",
            disabled=anova_table is None,
        )
    with dl2:
        st.download_button(
            label="Download post hoc results as CSV",
            data=pairwise_csv,
            file_name="posthoc_results.csv",
            mime="text/csv",
        )
