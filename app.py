from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

PAGE_TITLE = "LMM（線形混合効果モデル）解析アプリ"
PAGE_CAPTION = (
    "群（A-Dなど）×加熱（あり/なし）×時間（反復測定）のデータを、"
    "ランダム切片付き LMM で解析します。"
)
ALPHA_OPTIONS = [0.01, 0.05, 0.10]
CORRECTION_LABELS = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "FDR (Benjamini-Hochberg)",
}

SAMPLE_LONG_CSV = """sample_id,group,heat,time,value
1,A,加熱,直後,10.8
1,A,加熱,1週,10.4
1,A,加熱,2週,10.1
1,A,加熱,3週,9.9
1,A,加熱,4週,9.8
1,A,加熱,5週,9.6
2,A,加熱,直後,10.9
2,A,加熱,1週,10.5
2,A,加熱,2週,10.2
2,A,加熱,3週,10.0
2,A,加熱,4週,9.8
2,A,加熱,5週,9.7
1,A,非加熱,直後,10.7
1,A,非加熱,1週,10.6
1,A,非加熱,2週,10.5
1,A,非加熱,3週,10.4
1,A,非加熱,4週,10.4
1,A,非加熱,5週,10.3
1,B,加熱,直後,11.4
1,B,加熱,1週,10.9
1,B,加熱,2週,10.4
1,B,加熱,3週,10.1
1,B,加熱,4週,9.9
1,B,加熱,5週,9.7
1,B,非加熱,直後,11.3
1,B,非加熱,1週,11.1
1,B,非加熱,2週,10.9
1,B,非加熱,3週,10.8
1,B,非加熱,4週,10.7
1,B,非加熱,5週,10.6
"""

SAMPLE_WIDE_CSV = """sample_id,group,heat,直後,1週,2週,3週,4週,5週
1,A,加熱,10.8,10.4,10.1,9.9,9.8,9.6
2,A,加熱,10.9,10.5,10.2,10.0,9.8,9.7
1,A,非加熱,10.7,10.6,10.5,10.4,10.4,10.3
2,A,非加熱,10.8,10.7,10.6,10.5,10.5,10.4
1,B,加熱,11.4,10.9,10.4,10.1,9.9,9.7
2,B,加熱,11.2,10.8,10.5,10.2,10.0,9.8
1,B,非加熱,11.3,11.1,10.9,10.8,10.7,10.6
2,B,非加熱,11.1,11.0,10.8,10.7,10.6,10.5
"""


def build_blank_wide_template() -> bytes:
    rows: list[dict[str, object]] = []
    for group in ["A", "B", "C", "D"]:
        for heat in ["加熱", "非加熱"]:
            for sample_id in range(1, 11):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "group": group,
                        "heat": heat,
                        "直後": "",
                        "1週": "",
                        "2週": "",
                        "3週": "",
                        "4週": "",
                        "5週": "",
                    }
                )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def build_blank_long_template() -> bytes:
    rows: list[dict[str, object]] = []
    for group in ["A", "B", "C", "D"]:
        for heat in ["加熱", "非加熱"]:
            for sample_id in range(1, 11):
                for time in ["直後", "1週", "2週", "3週", "4週", "5週"]:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "group": group,
                            "heat": heat,
                            "time": time,
                            "value": "",
                        }
                    )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")



@dataclass
class Settings:
    alpha: float
    correction_method: str


@dataclass
class LMMFitResult:
    result: object | None
    formula: str
    fit_method: str
    notes: list[str]
    error: str
    terms_table: pd.DataFrame | None
    coefficients_table: pd.DataFrame | None


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.caption(PAGE_CAPTION)
    st.info(
        "このアプリは、同じサンプルを複数時点で追跡したデータを対象にしています。"
        " 主解析はランダム切片付き LMM です。"
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
    with st.expander("CSV形式の例（今回の実験デザイン向け）", expanded=True):
        st.markdown(
            """
**おすすめは wide形式** です。

今回の実験なら、**1行 = 1サンプル** にして、
列を `sample_id, group, heat, 直後, 1週, 2週, 3週, 4週, 5週` の順に並べると分かりやすいです。

`sample_id` は **各群×加熱条件の中で 1〜10 を繰り返してOK** です。
            """
        )

        st.markdown(
            """
**受け付ける形式**

- wide形式: 1行 = 1サンプル
- long形式: 1行 = 1サンプル×1時点
            """
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("wide形式の見本（推奨）")
            st.code(SAMPLE_WIDE_CSV, language="csv")
            st.download_button(
                label="wide形式の見本CSVをダウンロード",
                data=SAMPLE_WIDE_CSV.encode("utf-8-sig"),
                file_name="sample_lmm_wide.csv",
                mime="text/csv",
            )
            st.download_button(
                label="80サンプルの空テンプレート（wide）をダウンロード",
                data=build_blank_wide_template(),
                file_name="template_lmm_wide_80samples.csv",
                mime="text/csv",
            )

        with col_right:
            st.subheader("long形式の見本")
            st.code(SAMPLE_LONG_CSV, language="csv")
            st.download_button(
                label="long形式の見本CSVをダウンロード",
                data=SAMPLE_LONG_CSV.encode("utf-8-sig"),
                file_name="sample_lmm_long.csv",
                mime="text/csv",
            )
            st.download_button(
                label="80サンプルの空テンプレート（long）をダウンロード",
                data=build_blank_long_template(),
                file_name="template_lmm_long_80samples.csv",
                mime="text/csv",
            )

        st.caption(
            "wide形式では時間列の並び順がそのまま時系列順として使われます。"
            " 迷ったら wide形式テンプレートに数値を入れて使ってください。"
        )


def load_csv_flex(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception:
            continue
    raise ValueError("UTF-8 / UTF-8-SIG / CP932 のいずれでも読み込めませんでした。")


def ordered_unique(values: pd.Series) -> list[str]:
    ordered: list[str] = []
    for value in values.astype(str).tolist():
        if value not in ordered:
            ordered.append(value)
    return ordered


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_float(value) -> float:
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


def load_long_format(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("列の割り当て（long形式）")
    cols = df.columns.tolist()
    default_map = {name: name for name in ["sample_id", "group", "heat", "time", "value"] if name in cols}

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_col = st.selectbox("サンプルID列", options=cols, index=cols.index(default_map.get("sample_id", cols[0])))
        group_col = st.selectbox("群列", options=cols, index=cols.index(default_map.get("group", cols[min(1, len(cols)-1)])))
    with col2:
        heat_col = st.selectbox("加熱列", options=cols, index=cols.index(default_map.get("heat", cols[min(2, len(cols)-1)])))
        time_col = st.selectbox("時間列", options=cols, index=cols.index(default_map.get("time", cols[min(3, len(cols)-1)])))
    with col3:
        value_col = st.selectbox("測定値列", options=cols, index=cols.index(default_map.get("value", cols[-1])))

    work = df[[sample_col, group_col, heat_col, time_col, value_col]].copy()
    work.columns = ["sample_id_raw", "group", "heat", "time", "value"]
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value", "sample_id_raw", "group", "heat", "time"]).copy()
    return finalize_long_df(work)



def load_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("列の割り当て（wide形式）")
    cols = df.columns.tolist()
    non_time_defaults = [c for c in ["sample_id", "group", "heat"] if c in cols]

    col1, col2 = st.columns(2)
    with col1:
        sample_col = st.selectbox("サンプルID列", options=cols, index=cols.index(non_time_defaults[0]) if len(non_time_defaults) >= 1 else 0)
        group_col = st.selectbox("群列", options=cols, index=cols.index(non_time_defaults[1]) if len(non_time_defaults) >= 2 else min(1, len(cols)-1))
    with col2:
        heat_col = st.selectbox("加熱列", options=cols, index=cols.index(non_time_defaults[2]) if len(non_time_defaults) >= 3 else min(2, len(cols)-1))
        time_candidates = [c for c in cols if c not in {sample_col, group_col, heat_col}]
        default_times = time_candidates[:]
        time_cols = st.multiselect("時間列（左から時系列順）", options=time_candidates, default=default_times)

    if len(time_cols) < 2:
        st.warning("時間列を2列以上選んでください。")
        return pd.DataFrame()

    work = df[[sample_col, group_col, heat_col] + time_cols].copy()
    work.columns = ["sample_id_raw", "group", "heat"] + time_cols
    for column in time_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    long_df = work.melt(
        id_vars=["sample_id_raw", "group", "heat"],
        value_vars=time_cols,
        var_name="time",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value", "sample_id_raw", "group", "heat", "time"]).copy()
    return finalize_long_df(long_df, time_order=time_cols)



def finalize_long_df(long_df: pd.DataFrame, time_order: list[str] | None = None) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["sample_id_raw"] = long_df["sample_id_raw"].astype(str)
    long_df["group"] = long_df["group"].astype(str)
    long_df["heat"] = long_df["heat"].astype(str)
    long_df["time"] = long_df["time"].astype(str)
    long_df["subject_key"] = (
        long_df["group"].astype(str) + "|" + long_df["heat"].astype(str) + "|" + long_df["sample_id_raw"].astype(str)
    )

    if time_order is None:
        time_order = ordered_unique(long_df["time"])

    long_df["time"] = pd.Categorical(long_df["time"], categories=time_order, ordered=True)
    long_df = long_df.sort_values(["group", "heat", "subject_key", "time"]).reset_index(drop=True)
    return long_df



def validate_long_df(long_df: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    if long_df.empty:
        messages.append("有効なデータがありません。")
        return messages

    duplicated = long_df.duplicated(subset=["subject_key", "time"], keep=False)
    if duplicated.any():
        messages.append(
            "同じ subject_key と time の組み合わせが重複しています。"
            " 同一サンプル・同一時点は1行だけにしてください。"
        )

    if long_df["group"].nunique() < 2:
        messages.append("group が2水準未満です。")
    if long_df["heat"].nunique() < 2:
        messages.append("heat が2水準未満です。")
    if long_df["time"].nunique() < 2:
        messages.append("time が2水準未満です。")
    if long_df["subject_key"].nunique() < 4:
        messages.append("サンプル数が少なすぎます。少なくとも4サンプル程度は必要です。")

    complete_counts = long_df.groupby("subject_key")["time"].nunique()
    if not complete_counts.empty and complete_counts.min() < long_df["time"].nunique():
        messages.append(
            "欠測が含まれています。LMM自体は実行できますが、補助的な時点比較では一部のペア数が減ります。"
        )

    return messages



def summarize_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["group", "heat", "time"], observed=True)["value"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    grouped.columns = ["群", "加熱", "時間", "n", "平均", "標準偏差", "中央値", "最小値", "最大値"]
    return grouped



def fit_lmm(long_df: pd.DataFrame, alpha: float = 0.05) -> LMMFitResult:
    notes: list[str] = []
    error = ""
    fit_method = ""
    terms_table = None
    coefficients_table = None

    group_levels = ordered_unique(long_df["group"])
    heat_levels = ordered_unique(long_df["heat"])
    time_levels = [str(x) for x in long_df["time"].cat.categories.tolist()]

    formula = (
        f"value ~ C(group, levels={group_levels!r})"
        f" * C(heat, levels={heat_levels!r})"
        f" * C(time, levels={time_levels!r})"
    )

    work = long_df.copy()
    work["group"] = pd.Categorical(work["group"], categories=group_levels)
    work["heat"] = pd.Categorical(work["heat"], categories=heat_levels)
    work["time"] = pd.Categorical(work["time"].astype(str), categories=time_levels, ordered=True)

    methods = ["lbfgs", "bfgs", "powell", "cg"]
    fitted = None
    last_exception = None
    for method in methods:
        try:
            model = smf.mixedlm(
                formula=formula,
                data=work,
                groups=work["subject_key"],
                re_formula="1",
            )
            fitted = model.fit(reml=False, method=method, maxiter=500, disp=False)
            fit_method = method
            break
        except Exception as exc:
            last_exception = exc
            notes.append(f"{method} 法では収束しませんでした: {exc}")

    if fitted is None:
        error = f"LMM の適合に失敗しました: {last_exception}"
        return LMMFitResult(
            result=None,
            formula=formula,
            fit_method=fit_method,
            notes=notes,
            error=error,
            terms_table=None,
            coefficients_table=None,
        )

    try:
        wald = fitted.wald_test_terms(scalar=True)
        terms_table = wald.table.reset_index().rename(
            columns={
                "index": "項",
                "statistic": "χ²",
                "pvalue": "p値",
                "df_constraint": "自由度",
            }
        )
        terms_table["判定"] = terms_table["p値"].apply(
            lambda x: "有意" if pd.notna(x) and x < alpha else "NS"
        )
    except Exception as exc:
        notes.append(f"項ごとの Wald 検定表を作れませんでした: {exc}")

    try:
        coefficients_table = pd.DataFrame(
            {
                "係数": fitted.fe_params.index,
                "推定値": fitted.fe_params.values,
                "標準誤差": fitted.bse_fe.values,
                "z値": fitted.tvalues.loc[fitted.fe_params.index].values,
                "p値": fitted.pvalues.loc[fitted.fe_params.index].values,
            }
        )
    except Exception as exc:
        notes.append(f"固定効果係数表を作れませんでした: {exc}")

    return LMMFitResult(
        result=fitted,
        formula=formula,
        fit_method=fit_method,
        notes=notes,
        error=error,
        terms_table=terms_table,
        coefficients_table=coefficients_table,
    )



def create_profile_plot(long_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    summary = (
        long_df.groupby(["group", "heat", "time"], observed=True)["value"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    if summary.empty:
        return fig

    for (group, heat), part in summary.groupby(["group", "heat"], observed=True):
        means = part["mean"].to_numpy()
        counts = part["count"].to_numpy()
        sds = part["std"].fillna(0).to_numpy()
        sem = np.divide(sds, np.sqrt(np.maximum(counts, 1)), out=np.zeros_like(sds), where=np.maximum(counts, 1) > 0)
        x = np.arange(len(part))
        label = f"{group}-{heat}"
        ax.errorbar(x, means, yerr=sem, marker="o", capsize=3, label=label)

    time_labels = [str(x) for x in summary["time"].drop_duplicates().tolist()]
    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("時間")
    ax.set_ylabel("測定値")
    ax.set_title("群×加熱ごとの平均推移（エラーバーはSEM）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig



def create_spaghetti_subset_plot(long_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    subset_keys = long_df["subject_key"].drop_duplicates().tolist()[: min(30, long_df["subject_key"].nunique())]
    subset = long_df[long_df["subject_key"].isin(subset_keys)].copy()
    time_labels = [str(x) for x in long_df["time"].cat.categories.tolist()]
    xmap = {label: i for i, label in enumerate(time_labels)}

    for _, part in subset.groupby("subject_key"):
        xs = [xmap[str(x)] for x in part["time"].astype(str).tolist()]
        ax.plot(xs, part["value"].to_numpy(), marker="o", alpha=0.35)

    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("時間")
    ax.set_ylabel("測定値")
    ax.set_title("個体ごとの推移（先頭30サンプルまで）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig



def run_heat_posthoc(long_df: pd.DataFrame, alpha: float, correction_method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pvals: list[float] = []

    heat_levels = ordered_unique(long_df["heat"])
    if len(heat_levels) != 2:
        return pd.DataFrame()

    low_heat, high_heat = heat_levels[0], heat_levels[1]
    for (group, time), part in long_df.groupby(["group", "time"], observed=True):
        x = part.loc[part["heat"] == low_heat, "value"].dropna()
        y = part.loc[part["heat"] == high_heat, "value"].dropna()
        if len(x) < 2 or len(y) < 2:
            rows.append({
                "比較": f"{group} / {time}: {low_heat} vs {high_heat}",
                "n1": int(len(x)),
                "n2": int(len(y)),
                "平均差": np.nan,
                "未補正 p値": np.nan,
                "補正 p値": np.nan,
                "方法": "Welch t-test",
                "判定": "判定不可",
            })
            pvals.append(np.nan)
            continue
        test = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        p = safe_float(test.pvalue)
        rows.append({
            "比較": f"{group} / {time}: {low_heat} vs {high_heat}",
            "n1": int(len(x)),
            "n2": int(len(y)),
            "平均差": safe_float(x.mean() - y.mean()),
            "未補正 p値": p,
            "補正 p値": np.nan,
            "方法": "Welch t-test",
            "判定": "",
        })
        pvals.append(p)

    return apply_pvalue_correction(rows, pvals, alpha, correction_method)



def run_time_posthoc_against_baseline(long_df: pd.DataFrame, alpha: float, correction_method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pvals: list[float] = []
    time_levels = [str(x) for x in long_df["time"].cat.categories.tolist()]
    if len(time_levels) < 2:
        return pd.DataFrame()
    baseline = time_levels[0]

    for (group, heat), part in long_df.groupby(["group", "heat"], observed=True):
        wide = part.pivot_table(index="subject_key", columns="time", values="value", aggfunc="first", observed=False)
        for later in time_levels[1:]:
            if baseline not in wide.columns or later not in wide.columns:
                rows.append({
                    "比較": f"{group} / {heat}: {baseline} vs {later}",
                    "n": 0,
                    "平均差": np.nan,
                    "未補正 p値": np.nan,
                    "補正 p値": np.nan,
                    "方法": "paired t-test",
                    "判定": "判定不可",
                })
                pvals.append(np.nan)
                continue

            pair = wide[[baseline, later]].dropna()
            if len(pair) < 2:
                rows.append({
                    "比較": f"{group} / {heat}: {baseline} vs {later}",
                    "n": int(len(pair)),
                    "平均差": np.nan,
                    "未補正 p値": np.nan,
                    "補正 p値": np.nan,
                    "方法": "paired t-test",
                    "判定": "判定不可",
                })
                pvals.append(np.nan)
                continue

            test = stats.ttest_rel(pair[baseline], pair[later], nan_policy="omit")
            p = safe_float(test.pvalue)
            rows.append({
                "比較": f"{group} / {heat}: {baseline} vs {later}",
                "n": int(len(pair)),
                "平均差": safe_float(pair[baseline].mean() - pair[later].mean()),
                "未補正 p値": p,
                "補正 p値": np.nan,
                "方法": "paired t-test",
                "判定": "",
            })
            pvals.append(p)

    return apply_pvalue_correction(rows, pvals, alpha, correction_method)



def apply_pvalue_correction(
    rows: list[dict[str, object]],
    pvals: list[float],
    alpha: float,
    correction_method: str,
) -> pd.DataFrame:
    valid_indices = [i for i, p in enumerate(pvals) if pd.notna(p)]
    if valid_indices:
        valid_p = [pvals[i] for i in valid_indices]
        _, corrected, _, _ = multipletests(valid_p, alpha=alpha, method=correction_method)
        for idx, corrected_p in zip(valid_indices, corrected):
            rows[idx]["補正 p値"] = safe_float(corrected_p)
            rows[idx]["判定"] = "有意" if corrected_p < alpha else "NS"

    for row in rows:
        if row.get("判定", "") == "":
            row["判定"] = "判定不可"
    return pd.DataFrame(rows)



def render_dataset_overview(long_df: pd.DataFrame) -> None:
    st.subheader("解析に使う long 形式データ")
    st.dataframe(long_df.head(100), use_container_width=True)
    st.caption(
        f"行数: {len(long_df)} / サンプル数: {long_df['subject_key'].nunique()} / "
        f"群: {long_df['group'].nunique()} / 加熱: {long_df['heat'].nunique()} / 時間: {long_df['time'].nunique()}"
    )
    st.download_button(
        label="整形後 long 形式 CSV をダウンロード",
        data=to_csv_bytes(long_df),
        file_name="prepared_long_data.csv",
        mime="text/csv",
    )



def render_summary(long_df: pd.DataFrame) -> None:
    st.subheader("記述統計")
    summary_df = summarize_long_df(long_df)
    st.dataframe(summary_df, use_container_width=True)
    st.download_button(
        label="記述統計 CSV をダウンロード",
        data=to_csv_bytes(summary_df),
        file_name="descriptive_statistics.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = create_profile_plot(long_df)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = create_spaghetti_subset_plot(long_df)
        st.pyplot(fig)
        plt.close(fig)



def render_lmm_results(fit: LMMFitResult) -> None:
    st.subheader("LMM 主解析")
    st.code(fit.formula)

    if fit.error:
        st.error(fit.error)
        return

    st.success(f"LMM の適合に成功しました（method={fit.fit_method}）。")
    if fit.notes:
        for note in fit.notes:
            st.warning(note)

    if fit.terms_table is not None:
        st.markdown("**固定効果の全体検定（Wald χ²）**")
        st.dataframe(fit.terms_table, use_container_width=True)
        st.caption(
            "通常はまず `group:heat:time` の交互作用を見て、"
            "有意なら単純主効果や事後比較へ進みます。"
        )
        st.download_button(
            label="LMM 全体検定 CSV をダウンロード",
            data=to_csv_bytes(fit.terms_table),
            file_name="lmm_wald_terms.csv",
            mime="text/csv",
        )

    if fit.coefficients_table is not None:
        st.markdown("**固定効果係数**")
        st.dataframe(fit.coefficients_table, use_container_width=True)
        st.download_button(
            label="固定効果係数 CSV をダウンロード",
            data=to_csv_bytes(fit.coefficients_table),
            file_name="lmm_fixed_effects.csv",
            mime="text/csv",
        )



def render_posthoc(long_df: pd.DataFrame, settings: Settings) -> None:
    st.subheader("補助的な事後比較")
    st.caption(
        "ここは読みやすさを優先した補助解析です。主たる結論は上の LMM 主解析を優先してください。"
    )

    heat_df = run_heat_posthoc(long_df, settings.alpha, settings.correction_method)
    time_df = run_time_posthoc_against_baseline(long_df, settings.alpha, settings.correction_method)

    if not heat_df.empty:
        st.markdown("**各群×各時点での 加熱 vs 非加熱**")
        st.dataframe(heat_df, use_container_width=True)
        st.download_button(
            label="加熱 vs 非加熱 比較 CSV をダウンロード",
            data=to_csv_bytes(heat_df),
            file_name="posthoc_heat_within_group_time.csv",
            mime="text/csv",
        )

    if not time_df.empty:
        st.markdown("**各群×加熱条件での ベースライン時点 vs 各時点**")
        st.dataframe(time_df, use_container_width=True)
        st.download_button(
            label="ベースライン比較 CSV をダウンロード",
            data=to_csv_bytes(time_df),
            file_name="posthoc_time_vs_baseline.csv",
            mime="text/csv",
        )



def main() -> None:
    configure_page()
    settings = render_sidebar()
    render_samples()

    uploaded_file = st.file_uploader("CSV をアップロード", type=["csv"])
    if uploaded_file is None:
        st.info("CSV をアップロードすると解析を開始します。")
        return

    try:
        df = load_csv_flex(uploaded_file)
    except Exception as exc:
        st.error(f"CSV の読み込みに失敗しました: {exc}")
        return

    st.subheader("データプレビュー")
    st.dataframe(df.head(50), use_container_width=True)

    input_format = st.radio("入力形式", options=["long", "wide"], horizontal=True)
    if input_format == "long":
        long_df = load_long_format(df)
    else:
        long_df = load_wide_format(df)

    if long_df.empty:
        return

    messages = validate_long_df(long_df)
    errors = [msg for msg in messages if "重複" in msg or "未満" in msg or "ありません" in msg]
    infos = [msg for msg in messages if msg not in errors]

    for msg in infos:
        st.warning(msg)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    render_dataset_overview(long_df)
    render_summary(long_df)

    fit = fit_lmm(long_df, settings.alpha)
    render_lmm_results(fit)

    if not fit.error:
        render_posthoc(long_df, settings)


if __name__ == "__main__":
    main()
