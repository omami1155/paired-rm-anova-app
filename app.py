from __future__ import annotations

import io
import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

PAGE_TITLE = "対応のある検定と反復測定 ANOVA"
PAGE_CAPTION = (
    "2条件では Shapiro-Wilk, paired t-test, Wilcoxon, "
    "3条件以上では反復測定 ANOVA と多重比較を実行します。"
)
ALPHA_OPTIONS = [0.01, 0.05, 0.10]
CORRECTION_LABELS = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "FDR (Benjamini-Hochberg)",
}
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


@dataclass
class Settings:
    alpha: float
    correction_method: str


@dataclass
class PairedAnalysisResult:
    results_df: pd.DataFrame
    clean_df: pd.DataFrame
    diff: pd.Series
    primary_test: str
    shapiro_p: float
    primary_p: float
    excluded_rows: int


@dataclass
class RMAnovaResult:
    anova_table: pd.DataFrame | None
    pairwise_df: pd.DataFrame
    complete_df: pd.DataFrame
    long_df: pd.DataFrame
    anova_note: str
    partial_eta_squared: float
    excluded_rows: int


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.caption(PAGE_CAPTION)
    st.info(
        "列が条件、行がサンプルです。"
        "2条件なら対応のある検定、3条件以上なら反復測定 ANOVA を実行します。"
    )


def render_sidebar() -> Settings:
    st.sidebar.header("設定")
    alpha = st.sidebar.selectbox(
        "有意水準 α",
        options=ALPHA_OPTIONS,
        index=1,
        format_func=lambda value: f"{value:.2f}",
    )
    correction_method = st.sidebar.selectbox(
        "多重比較の補正方法",
        options=list(CORRECTION_LABELS),
        index=0,
        format_func=lambda key: CORRECTION_LABELS[key],
    )
    return Settings(alpha=alpha, correction_method=correction_method)


def render_samples() -> None:
    with st.expander("CSV形式の例", expanded=True):
        st.markdown(
            "列が条件、行がサンプルです。ヘッダー付きのCSVを読み込みます。"
        )
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**2条件の例**")
            st.code(SAMPLE_TWO_CSV, language="csv")
            st.download_button(
                label="サンプルCSVをダウンロード",
                data=SAMPLE_TWO_CSV.encode("utf-8-sig"),
                file_name="sample_paired_two_conditions.csv",
                mime="text/csv",
            )
        with col_right:
            st.markdown("**3条件以上の例**")
            st.code(SAMPLE_THREE_CSV, language="csv")
            st.download_button(
                label="サンプルCSVをダウンロード",
                data=SAMPLE_THREE_CSV.encode("utf-8-sig"),
                file_name="sample_repeated_measures.csv",
                mime="text/csv",
            )


def load_csv_flex(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception:
            continue
    raise ValueError("UTF-8 / UTF-8-SIG / CP932 のいずれでも読み込めませんでした。")


def coerce_numeric_columns(
    source_df: pd.DataFrame, selected_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    numeric_df = source_df[selected_cols].copy()
    dropped_counts: dict[str, int] = {}

    for column in selected_cols:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
        original_non_null = int(source_df[column].notna().sum())
        numeric_non_null = int(numeric_df[column].notna().sum())
        dropped_counts[column] = original_non_null - numeric_non_null

    return numeric_df, dropped_counts


def summarize_series(series: pd.Series, label: str) -> dict[str, float | int | str]:
    clean = pd.Series(series).dropna()
    q1 = clean.quantile(0.25) if not clean.empty else np.nan
    q3 = clean.quantile(0.75) if not clean.empty else np.nan

    return {
        "グループ": label,
        "n数": int(clean.size),
        "平均": float(clean.mean()) if clean.size else np.nan,
        "標準偏差": float(clean.std(ddof=1)) if clean.size >= 2 else np.nan,
        "最小値": float(clean.min()) if clean.size else np.nan,
        "Q1": float(q1) if clean.size else np.nan,
        "中央値": float(clean.median()) if clean.size else np.nan,
        "Q3": float(q3) if clean.size else np.nan,
        "最大値": float(clean.max()) if clean.size else np.nan,
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([summarize_series(df[column], column) for column in df.columns])


def safe_shapiro(values: pd.Series) -> tuple[float, str]:
    clean = pd.Series(values).dropna()
    if clean.size < 3:
        return np.nan, "差分が 3 件未満のため Shapiro-Wilk を実行できません。"
    if clean.size > 5000:
        return np.nan, "差分が 5000 件を超えるため Shapiro-Wilk の前提外です。"
    if clean.nunique() < 2:
        return np.nan, "差分がすべて同一のため正規性を評価できません。"

    try:
        result = stats.shapiro(clean)
        return float(result.pvalue), ""
    except Exception as exc:
        return np.nan, f"Shapiro-Wilk の実行に失敗しました: {exc}"


def interpret_normality(pvalue: float, alpha: float) -> str:
    if pd.isna(pvalue):
        return "判定不可"
    if pvalue < alpha:
        return "正規性を仮定しにくい"
    return "正規性を仮定してよい可能性が高い"


def interpret_difference(pvalue: float, alpha: float) -> str:
    if pd.isna(pvalue):
        return "判定不可"
    if pvalue < alpha:
        return "有意差あり"
    return "有意差を示す十分な根拠なし"


def safe_float(value: float | np.floating | None) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def format_pvalue(pvalue: float) -> str:
    if pd.isna(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< 0.001"
    return f"{pvalue:.4f}"


def cohen_dz(diff: pd.Series) -> tuple[float, str]:
    clean = pd.Series(diff).dropna()
    if clean.size < 2:
        return np.nan, "効果量の算出には 2 ペア以上が必要です。"

    sd = clean.std(ddof=1)
    if pd.isna(sd) or np.isclose(sd, 0.0):
        return np.nan, "差分の標準偏差が 0 のため Cohen's dz を算出できません。"

    return float(clean.mean() / sd), ""


def rank_biserial_correlation(diff: pd.Series) -> tuple[float, str]:
    clean = pd.Series(diff).dropna()
    non_zero = clean[~np.isclose(clean, 0.0)]
    if non_zero.empty:
        return np.nan, "差分がすべて 0 のため rank-biserial correlation を算出できません。"

    ranks = stats.rankdata(np.abs(non_zero))
    positive_rank_sum = float(ranks[non_zero > 0].sum())
    negative_rank_sum = float(ranks[non_zero < 0].sum())
    total_rank_sum = float(len(non_zero) * (len(non_zero) + 1) / 2)
    effect_size = (positive_rank_sum - negative_rank_sum) / total_rank_sum
    return float(effect_size), ""


def build_paired_results_rows(
    col_left: str,
    col_right: str,
    clean_df: pd.DataFrame,
    alpha: float,
    primary_test: str,
    shapiro_p: float,
    shapiro_note: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    rows.append(
        {
            "区分": "前提確認",
            "検定": "Shapiro-Wilk（差分）",
            "p値": safe_float(shapiro_p),
            "α": alpha,
            "推奨": "",
            "解釈": interpret_normality(shapiro_p, alpha),
            "補足": shapiro_note or "差分の正規性を確認しました。",
        }
    )

    if clean_df.shape[0] >= 2:
        try:
            t_result = stats.ttest_rel(clean_df[col_left], clean_df[col_right], nan_policy="omit")
            rows.append(
                {
                    "区分": "主解析候補",
                    "検定": "paired t-test",
                    "p値": safe_float(t_result.pvalue),
                    "α": alpha,
                    "推奨": "〇" if primary_test == "paired t-test" else "",
                    "解釈": interpret_difference(safe_float(t_result.pvalue), alpha),
                    "補足": "平均差に基づく検定です。",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "区分": "主解析候補",
                    "検定": "paired t-test",
                    "p値": np.nan,
                    "α": alpha,
                    "推奨": "〇" if primary_test == "paired t-test" else "",
                    "解釈": "判定不可",
                    "補足": f"paired t-test の実行に失敗しました: {exc}",
                }
            )
    else:
        rows.append(
            {
                "区分": "主解析候補",
                "検定": "paired t-test",
                "p値": np.nan,
                "α": alpha,
                "推奨": "〇" if primary_test == "paired t-test" else "",
                "解釈": "判定不可",
                "補足": "paired t-test には 2 ペア以上の完全データが必要です。",
            }
        )

    if clean_df.shape[0] >= 1:
        try:
            wilcoxon_result = stats.wilcoxon(
                clean_df[col_left],
                clean_df[col_right],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            rows.append(
                {
                    "区分": "主解析候補",
                    "検定": "Wilcoxon signed-rank",
                    "p値": safe_float(wilcoxon_result.pvalue),
                    "α": alpha,
                    "推奨": "〇" if primary_test == "Wilcoxon signed-rank" else "",
                    "解釈": interpret_difference(safe_float(wilcoxon_result.pvalue), alpha),
                    "補足": "順位和に基づくノンパラメトリック検定です。",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "区分": "主解析候補",
                    "検定": "Wilcoxon signed-rank",
                    "p値": np.nan,
                    "α": alpha,
                    "推奨": "〇" if primary_test == "Wilcoxon signed-rank" else "",
                    "解釈": "判定不可",
                    "補足": f"Wilcoxon の実行に失敗しました: {exc}",
                }
            )
    else:
        rows.append(
            {
                "区分": "主解析候補",
                "検定": "Wilcoxon signed-rank",
                "p値": np.nan,
                "α": alpha,
                "推奨": "〇" if primary_test == "Wilcoxon signed-rank" else "",
                "解釈": "判定不可",
                "補足": "Wilcoxon には 1 ペア以上の完全データが必要です。",
            }
        )

    return rows


def run_paired_analysis(df: pd.DataFrame, alpha: float) -> PairedAnalysisResult:
    col_left, col_right = df.columns.tolist()
    clean_df = df[[col_left, col_right]].dropna().copy()
    diff = clean_df[col_left] - clean_df[col_right]
    shapiro_p, shapiro_note = safe_shapiro(diff)
    primary_test = "paired t-test" if pd.notna(shapiro_p) and shapiro_p >= alpha else "Wilcoxon signed-rank"

    rows = build_paired_results_rows(
        col_left=col_left,
        col_right=col_right,
        clean_df=clean_df,
        alpha=alpha,
        primary_test=primary_test,
        shapiro_p=shapiro_p,
        shapiro_note=shapiro_note,
    )
    results_df = pd.DataFrame(rows)

    primary_row = results_df.loc[results_df["検定"] == primary_test, "p値"]
    primary_p = safe_float(primary_row.iloc[0]) if not primary_row.empty else np.nan

    return PairedAnalysisResult(
        results_df=results_df,
        clean_df=clean_df,
        diff=diff,
        primary_test=primary_test,
        shapiro_p=shapiro_p,
        primary_p=primary_p,
        excluded_rows=int(df.shape[0] - clean_df.shape[0]),
    )


def wide_to_long_complete(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete_df = df.dropna().reset_index(drop=True).copy()
    long_df = (
        complete_df.assign(subject=np.arange(1, len(complete_df) + 1))
        .melt(id_vars="subject", var_name="condition", value_name="value")
    )
    return complete_df, long_df


def compute_partial_eta_squared(anova_table: pd.DataFrame | None) -> float:
    if anova_table is None or anova_table.empty:
        return np.nan
    required_columns = {"F Value", "Num DF", "Den DF"}
    if not required_columns.issubset(anova_table.columns):
        return np.nan

    row = anova_table.iloc[0]
    numerator = row["F Value"] * row["Num DF"]
    denominator = numerator + row["Den DF"]
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def build_pairwise_table(
    complete_df: pd.DataFrame,
    alpha: float,
    correction_method: str,
) -> pd.DataFrame:
    comparisons: list[dict[str, float | int | str]] = []
    raw_pvalues: list[float] = []

    for col_left, col_right in itertools.combinations(complete_df.columns.tolist(), 2):
        pair_diff = complete_df[col_left] - complete_df[col_right]
        effect_size, effect_note = cohen_dz(pair_diff)

        if complete_df.shape[0] < 2:
            comparisons.append(
                {
                    "比較": f"{col_left} vs {col_right}",
                    "n": int(complete_df.shape[0]),
                    "平均差": np.nan,
                    "未補正 p値": np.nan,
                    "補正 p値": np.nan,
                    "効果量": effect_size,
                    "効果量名": "Cohen's dz",
                    "解釈": "判定不可",
                    "補足": "対応のある t 検定には 2 件以上の完全データが必要です。",
                }
            )
            raw_pvalues.append(np.nan)
            continue

        try:
            test_result = stats.ttest_rel(complete_df[col_left], complete_df[col_right])
            raw_pvalue = safe_float(test_result.pvalue)
            comparisons.append(
                {
                    "比較": f"{col_left} vs {col_right}",
                    "n": int(complete_df.shape[0]),
                    "平均差": safe_float(pair_diff.mean()),
                    "未補正 p値": raw_pvalue,
                    "補正 p値": np.nan,
                    "効果量": effect_size,
                    "効果量名": "Cohen's dz",
                    "解釈": "",
                    "補足": effect_note or "対応のある t 検定を実行しました。",
                }
            )
            raw_pvalues.append(raw_pvalue)
        except Exception as exc:
            comparisons.append(
                {
                    "比較": f"{col_left} vs {col_right}",
                    "n": int(complete_df.shape[0]),
                    "平均差": np.nan,
                    "未補正 p値": np.nan,
                    "補正 p値": np.nan,
                    "効果量": effect_size,
                    "効果量名": "Cohen's dz",
                    "解釈": "判定不可",
                    "補足": f"対応のある t 検定の実行に失敗しました: {exc}",
                }
            )
            raw_pvalues.append(np.nan)

    valid_mask = [pd.notna(value) for value in raw_pvalues]
    if any(valid_mask):
        valid_pvalues = [value for value in raw_pvalues if pd.notna(value)]
        reject, corrected, _, _ = multipletests(
            valid_pvalues,
            alpha=alpha,
            method=correction_method,
        )
        corrected_index = 0
        for row_index, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            comparisons[row_index]["補正 p値"] = safe_float(corrected[corrected_index])
            comparisons[row_index]["解釈"] = interpret_difference(
                safe_float(corrected[corrected_index]),
                alpha,
            )
            comparisons[row_index]["有意"] = "Yes" if reject[corrected_index] else "No"
            corrected_index += 1

    for row in comparisons:
        row.setdefault("有意", "")
        row.setdefault("解釈", "判定不可")

    return pd.DataFrame(comparisons)


def run_rm_anova_analysis(
    df: pd.DataFrame,
    alpha: float,
    correction_method: str,
) -> RMAnovaResult:
    complete_df, long_df = wide_to_long_complete(df)
    anova_table: pd.DataFrame | None = None
    anova_note = ""

    if complete_df.shape[0] >= 2:
        try:
            fitted = AnovaRM(
                data=long_df,
                depvar="value",
                subject="subject",
                within=["condition"],
            ).fit()
            anova_table = fitted.anova_table.reset_index().rename(columns={"index": "効果"})
        except Exception as exc:
            anova_note = f"反復測定 ANOVA の実行に失敗しました: {exc}"
    else:
        anova_note = "反復測定 ANOVA には 2 名以上の完全データが必要です。"

    partial_eta_squared = compute_partial_eta_squared(anova_table)
    if anova_table is not None:
        anova_table["partial eta^2"] = partial_eta_squared

    return RMAnovaResult(
        anova_table=anova_table,
        pairwise_df=build_pairwise_table(complete_df, alpha, correction_method),
        complete_df=complete_df,
        long_df=long_df,
        anova_note=anova_note,
        partial_eta_squared=partial_eta_squared,
        excluded_rows=int(df.shape[0] - complete_df.shape[0]),
    )


def create_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(5.5, len(df.columns) * 1.4), 4.0))
    values = [df[column].dropna() for column in df.columns]
    ax.boxplot(values, tick_labels=df.columns.tolist())
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_histogram(diff: pd.Series, label_left: str, label_right: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.hist(diff.dropna(), bins="auto", color="#3B82F6", alpha=0.85, edgecolor="white")
    ax.set_xlabel(f"Difference ({label_left} - {label_right})")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_qq_plot(diff: pd.Series):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    clean = diff.dropna()
    if clean.size >= 2:
        stats.probplot(clean, dist="norm", plot=ax)
        ax.set_title("")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def create_spaghetti_plot(complete_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(5.5, len(complete_df.columns) * 1.4), 4.0))
    x_positions = np.arange(len(complete_df.columns))
    for _, row in complete_df.iterrows():
        ax.plot(x_positions, row.values, marker="o", alpha=0.45, color="#64748B")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(complete_df.columns.tolist())
    ax.set_ylabel("Value")
    ax.set_title("Subject-level trajectories")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_profile_plot(complete_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(5.5, len(complete_df.columns) * 1.4), 4.0))
    means = complete_df.mean(axis=0)
    sample_size = int(complete_df.shape[0])
    x_positions = np.arange(len(complete_df.columns))

    if sample_size >= 2:
        sem = complete_df.sem(axis=0, ddof=1)
        critical_t = stats.t.ppf(0.975, sample_size - 1)
        error = sem * critical_t
    else:
        error = pd.Series(np.zeros(len(complete_df.columns)), index=complete_df.columns)

    ax.errorbar(
        x_positions,
        means.values,
        yerr=error.values,
        fmt="-o",
        capsize=4,
        linewidth=2,
        color="#0F766E",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(complete_df.columns.tolist())
    ax.set_ylabel("Mean value")
    ax.set_title("Condition means with 95% CI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def render_data_quality_warning(dropped_counts: dict[str, int]) -> None:
    if any(value > 0 for value in dropped_counts.values()):
        detail = ", ".join(f"{column}: {count}" for column, count in dropped_counts.items() if count > 0)
        st.warning(f"数値に変換できない値を除外しました。列ごとの件数: {detail}")


def render_summary_section(selected_df: pd.DataFrame) -> None:
    st.subheader("記述統計")
    st.dataframe(build_summary_table(selected_df), use_container_width=True)
    figure = create_boxplot(selected_df)
    st.pyplot(figure)
    plt.close(figure)


def render_paired_results(result: PairedAnalysisResult, alpha: float) -> None:
    st.subheader("対応のある 2 群比較")

    st.dataframe(result.results_df, use_container_width=True)

    st.markdown("#### 解釈メモ")
    if pd.notna(result.shapiro_p) and result.shapiro_p >= alpha:
        st.info(
            "差分の正規性を棄却する十分な根拠はなく、対応のある t 検定の前提を大きく損なう所見もみられないため、"
            "このデータでは paired t-test を第一候補として扱いました。"
        )
    elif pd.notna(result.shapiro_p):
        st.info(
            "差分の正規性を仮定しにくい結果だったため、このデータでは正規性の仮定により依存しにくい "
            "Wilcoxon signed-rank を第一候補として扱いました。"
        )
    else:
        st.info(
            "差分の正規性を十分に判定できなかったため、このデータではより保守的に "
            "Wilcoxon signed-rank を第一候補として扱いました。"
        )

    st.markdown("#### QQプロット")
    qq_figure = create_qq_plot(result.diff)
    st.pyplot(qq_figure)
    plt.close(qq_figure)

    st.markdown("#### 解析に使ったデータ")
    export_df = result.clean_df.copy()
    export_df["difference"] = result.diff.values
    st.dataframe(export_df, use_container_width=True)
    st.download_button(
        label="結果 CSV をダウンロード",
        data=to_csv_bytes(result.results_df),
        file_name="paired_test_results.csv",
        mime="text/csv",
    )


def render_rm_anova_results(result: RMAnovaResult, correction_method: str) -> None:
    st.subheader("反復測定 ANOVA")

    metrics = st.columns(4)
    metrics[0].metric("完全ケース", int(result.complete_df.shape[0]))
    metrics[1].metric("除外行", int(result.excluded_rows))
    metrics[2].metric("条件数", int(result.complete_df.shape[1]))
    metrics[3].metric("partial eta^2", "NA" if pd.isna(result.partial_eta_squared) else f"{result.partial_eta_squared:.3f}")

    figure_left, figure_right = st.columns(2)
    with figure_left:
        spaghetti_figure = create_spaghetti_plot(result.complete_df)
        st.pyplot(spaghetti_figure)
        plt.close(spaghetti_figure)
    with figure_right:
        profile_figure = create_profile_plot(result.complete_df)
        st.pyplot(profile_figure)
        plt.close(profile_figure)

    if result.anova_note:
        st.error(result.anova_note)
    else:
        st.markdown("**ANOVA 結果**")
        st.dataframe(result.anova_table, use_container_width=True)

    st.markdown(f"**事後比較（{CORRECTION_LABELS[correction_method]} 補正）**")
    st.dataframe(result.pairwise_df, use_container_width=True)
    st.info(
        "反復測定 ANOVA と事後比較は、選択したすべての条件で欠損のない完全ケースのみで計算しています。"
    )

    tab_notes, tab_data = st.tabs(["解釈", "解析に使ったデータ"])
    with tab_notes:
        st.markdown(
            "事後比較は対応のある t 検定を全組み合わせで実行し、選択した補正方法で p 値を補正しています。"
        )
    with tab_data:
        st.markdown("**完全ケース（wide 形式）**")
        st.dataframe(result.complete_df, use_container_width=True)
        st.markdown("**long 形式**")
        st.dataframe(result.long_df, use_container_width=True)

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            label="ANOVA 結果 CSV をダウンロード",
            data=to_csv_bytes(result.anova_table) if result.anova_table is not None else b"",
            file_name="repeated_measures_anova_results.csv",
            mime="text/csv",
            disabled=result.anova_table is None,
        )
    with download_right:
        st.download_button(
            label="事後比較 CSV をダウンロード",
            data=to_csv_bytes(result.pairwise_df),
            file_name="posthoc_results.csv",
            mime="text/csv",
        )


def main() -> None:
    configure_page()
    settings = render_sidebar()
    render_samples()

    uploaded_file = st.file_uploader("wide 形式の CSV をアップロード", type=["csv"])
    if uploaded_file is None:
        st.info("CSV をアップロードすると解析を開始します。")
        return

    try:
        df = load_csv_flex(uploaded_file)
    except Exception as exc:
        st.error(f"CSV の読み込みに失敗しました: {exc}")
        return

    if df.shape[1] < 2:
        st.error("解析には 2 列以上が必要です。")
        return

    st.subheader("データプレビュー")
    st.dataframe(df.head(30), use_container_width=True)

    selected_cols = st.multiselect(
        "解析に使う列を選択してください",
        options=df.columns.tolist(),
        default=df.columns.tolist()[: min(3, len(df.columns))],
    )

    if len(selected_cols) < 2:
        st.warning("少なくとも 2 列を選択してください。")
        return

    selected_df, dropped_counts = coerce_numeric_columns(df, selected_cols)
    render_data_quality_warning(dropped_counts)
    render_summary_section(selected_df)

    if len(selected_cols) == 2:
        result = run_paired_analysis(selected_df, settings.alpha)
        render_paired_results(result, settings.alpha)
        return

    result = run_rm_anova_analysis(
        df=selected_df,
        alpha=settings.alpha,
        correction_method=settings.correction_method,
    )
    render_rm_anova_results(result, settings.correction_method)


if __name__ == "__main__":
    main()
