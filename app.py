import io
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="対応のある検定・反復測定ANOVA", layout="wide")
st.title("対応のある検定・反復測定ANOVA")
st.caption("2条件: Shapiro-Wilk(差) / paired t-test / Wilcoxon, 3条件以上: 反復測定ANOVA / 事後比較")
st.info("同一対象を複数条件で測定した wide 形式CSV を想定しています。各行が1対象、各列が1条件です。")

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
    encodings = ["utf-8", "utf-8-sig", "cp932"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def to_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def summarize(series, name):
    x = pd.Series(series).dropna()
    q1 = x.quantile(0.25) if len(x) > 0 else np.nan
    q3 = x.quantile(0.75) if len(x) > 0 else np.nan
    return {
        "条件": name,
        "n数": int(x.shape[0]),
        "平均": float(x.mean()) if len(x) else np.nan,
        "標準偏差": float(x.std(ddof=1)) if len(x) >= 2 else np.nan,
        "最小値": float(x.min()) if len(x) else np.nan,
        "Q1": float(q1) if len(x) else np.nan,
        "中央値": float(x.median()) if len(x) else np.nan,
        "Q3": float(q3) if len(x) else np.nan,
        "最大値": float(x.max()) if len(x) else np.nan,
    }


def safe_shapiro(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, "n<3のため実行不可"
    if len(x) > 5000:
        return np.nan, "n>5000のため Shapiro-Wilk は実行対象外"
    if x.nunique() < 2:
        return np.nan, "差がほぼ一定のため実行不可"
    try:
        res = stats.shapiro(x)
        return float(res.pvalue), ""
    except Exception as e:
        return np.nan, f"実行不可: {e}"


def interpret_shapiro(p, alpha):
    if pd.isna(p):
        return "判定不可"
    return "正規性を仮定しにくい" if p < alpha else "正規性を棄却する十分な根拠なし"


def interpret_difference(p, alpha):
    if pd.isna(p):
        return "判定不可"
    return "有意差あり" if p < alpha else "有意差を示す十分な根拠なし"


def add_result(rows, category, test_name, pvalue, alpha, interpretation, note="", primary=False):
    rows.append({
        "区分": category,
        "検定": test_name,
        "p値": pvalue,
        "α": alpha,
        "推奨": "○" if primary else "",
        "解釈": interpretation,
        "備考": note if note else "特記事項なし",
    })


def run_paired_tests(df_two, alpha=0.05):
    rows = []
    col1, col2 = df_two.columns.tolist()
    clean = df_two[[col1, col2]].dropna().copy()
    x = clean[col1]
    y = clean[col2]
    diff = x - y

    shapiro_p, shapiro_note = safe_shapiro(diff)
    add_result(
        rows,
        "前提確認",
        "Shapiro-Wilk（差分）",
        shapiro_p,
        alpha,
        interpret_shapiro(shapiro_p, alpha),
        shapiro_note,
    )

    primary_test = "paired t-test" if pd.notna(shapiro_p) and shapiro_p >= alpha else "Wilcoxon signed-rank"

    if len(clean) >= 2:
        try:
            t_res = stats.ttest_rel(x, y, alternative="two-sided", nan_policy="omit")
            add_result(
                rows,
                "群比較",
                "paired t-test",
                float(t_res.pvalue),
                alpha,
                interpret_difference(float(t_res.pvalue), alpha),
                primary=(primary_test == "paired t-test"),
            )
        except Exception as e:
            add_result(rows, "群比較", "paired t-test", np.nan, alpha, "判定不可", f"実行不可: {e}", primary=(primary_test == "paired t-test"))
    else:
        add_result(rows, "群比較", "paired t-test", np.nan, alpha, "判定不可", "対応のある完全データが2組以上必要", primary=(primary_test == "paired t-test"))

    if len(clean) >= 1:
        try:
            w_res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            add_result(
                rows,
                "群比較",
                "Wilcoxon signed-rank",
                float(w_res.pvalue),
                alpha,
                interpret_difference(float(w_res.pvalue), alpha),
                primary=(primary_test == "Wilcoxon signed-rank"),
            )
        except Exception as e:
            add_result(rows, "群比較", "Wilcoxon signed-rank", np.nan, alpha, "判定不可", f"実行不可: {e}", primary=(primary_test == "Wilcoxon signed-rank"))
    else:
        add_result(rows, "群比較", "Wilcoxon signed-rank", np.nan, alpha, "判定不可", "対応のある完全データが必要", primary=(primary_test == "Wilcoxon signed-rank"))

    return pd.DataFrame(rows), clean, diff, primary_test, shapiro_p


def wide_to_long_complete(df_wide):
    complete = df_wide.dropna().copy().reset_index(drop=True)
    complete.insert(0, "subject", np.arange(1, len(complete) + 1))
    long_df = complete.melt(id_vars="subject", var_name="condition", value_name="value")
    return complete, long_df


def run_rm_anova(df_selected, alpha=0.05, correction_method="holm"):
    complete, long_df = wide_to_long_complete(df_selected)

    anova_table = None
    anova_note = ""
    try:
        model = AnovaRM(data=long_df, depvar="value", subject="subject", within=["condition"])
        fitted = model.fit()
        anova_table = fitted.anova_table.reset_index().rename(columns={"index": "効果"})
    except Exception as e:
        anova_note = f"反復測定ANOVAを実行できませんでした: {e}"

    pairwise_rows = []
    pairs = list(itertools.combinations(df_selected.columns.tolist(), 2))
    raw_pvals = []
    interim = []

    for a, b in pairs:
        pair = complete[[a, b]].dropna()
        if len(pair) < 2:
            interim.append((a, b, len(pair), np.nan, np.nan, "対応のある完全データが2組以上必要"))
            raw_pvals.append(np.nan)
            continue
        try:
            res = stats.ttest_rel(pair[a], pair[b], alternative="two-sided")
            p = float(res.pvalue)
            raw_pvals.append(p)
            interim.append((a, b, len(pair), float(pair[a].mean() - pair[b].mean()), p, ""))
        except Exception as e:
            interim.append((a, b, len(pair), np.nan, np.nan, f"実行不可: {e}"))
            raw_pvals.append(np.nan)

    valid_mask = [pd.notna(p) for p in raw_pvals]
    corrected = [np.nan] * len(raw_pvals)
    reject = [False] * len(raw_pvals)
    if any(valid_mask):
        valid_pvals = [p for p in raw_pvals if pd.notna(p)]
        rej, corr_p, _, _ = multipletests(valid_pvals, alpha=alpha, method=correction_method)
        j = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                corrected[i] = float(corr_p[j])
                reject[i] = bool(rej[j])
                j += 1

    for i, item in enumerate(interim):
        a, b, n_pair, mean_diff, raw_p, note = item
        p_corr = corrected[i]
        pairwise_rows.append({
            "比較": f"{a} vs {b}",
            "n数": n_pair,
            "平均差({}-{})".format(a, b): mean_diff,
            "未補正p値": raw_p,
            f"補正p値({correction_method})": p_corr,
            "解釈": interpret_difference(p_corr, alpha) if pd.notna(p_corr) else "判定不可",
            "備考": note if note else "特記事項なし",
        })

    return anova_table, pd.DataFrame(pairwise_rows), complete, long_df, anova_note


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
    ax.set_title("各対象の推移")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


st.sidebar.header("設定")
alpha = st.sidebar.selectbox("有意水準 α", options=ALPHA_OPTIONS, index=1, format_func=lambda x: f"{x:.2f}")
correction_method = st.sidebar.selectbox(
    "多重比較補正（3条件以上の事後比較）",
    options=["holm", "bonferroni", "fdr_bh"],
    index=0,
)

with st.expander("CSV形式の例", expanded=True):
    st.markdown("各行が同一対象、各列が条件です。欠損がある行は、選択した条件の解析から除外されます。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**2条件の例**")
        st.code(SAMPLE_TWO_CSV, language="csv")
        st.download_button(
            label="2条件サンプルCSVをダウンロード",
            data=SAMPLE_TWO_CSV.encode("utf-8-sig"),
            file_name="sample_paired_two_conditions.csv",
            mime="text/csv",
        )
    with c2:
        st.markdown("**3条件以上の例**")
        st.code(SAMPLE_THREE_CSV, language="csv")
        st.download_button(
            label="3条件サンプルCSVをダウンロード",
            data=SAMPLE_THREE_CSV.encode("utf-8-sig"),
            file_name="sample_repeated_measures.csv",
            mime="text/csv",
        )

uploaded_file = st.file_uploader("wide形式CSVをアップロード", type=["csv"])

if uploaded_file is None:
    st.info("CSVをアップロードしてください。")
    st.stop()

try:
    df = load_csv_flex(uploaded_file)
except Exception as e:
    st.error(f"CSVの読み込みに失敗しました: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("列が2本以上必要です。")
    st.stop()

st.subheader("読み込んだデータ")
st.dataframe(df.head(30), use_container_width=True)

selected_cols = st.multiselect(
    "解析に使う条件列を選択してください（2列なら対応のある検定、3列以上なら反復測定ANOVA）",
    options=df.columns.tolist(),
    default=df.columns.tolist()[: min(3, len(df.columns))],
)

if len(selected_cols) < 2:
    st.warning("少なくとも2列選択してください。")
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
    msg = ", ".join([f"{k}: {v}件" for k, v in non_numeric_counts.items() if v > 0])
    st.warning(f"数値変換できない値を除外しました（{msg}）。")

st.subheader("記述統計")
summary_df = pd.DataFrame([summarize(selected_df[col], col) for col in selected_cols])
st.dataframe(summary_df, use_container_width=True)

st.subheader("箱ひげ図")
fig_box = plot_box(selected_df)
st.pyplot(fig_box)
plt.close(fig_box)

if len(selected_cols) == 2:
    st.subheader("対応のある2群比較")
    results_df, clean_df, diff, primary_test, shapiro_p = run_paired_tests(selected_df, alpha=alpha)

    st.write(f"完全データのペア数: **{len(clean_df)}**")
    st.dataframe(results_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**差分のヒストグラム**")
        fig_hist, ax_hist = plt.subplots(figsize=(4.5, 3.2))
        ax_hist.hist(diff.dropna(), bins="auto")
        ax_hist.set_xlabel(f"差分 ({selected_cols[0]} - {selected_cols[1]})")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with c2:
        st.markdown("**差分のQQプロット**")
        fig_qq, ax_qq = plt.subplots(figsize=(4.5, 3.2))
        if diff.dropna().shape[0] >= 2:
            stats.probplot(diff.dropna(), dist="norm", plot=ax_qq)
            ax_qq.grid(True, alpha=0.3)
        else:
            ax_qq.text(0.5, 0.5, "データ不足", ha="center", va="center")
            ax_qq.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_qq)
        plt.close(fig_qq)

    if pd.notna(shapiro_p) and shapiro_p >= alpha:
        st.info("差分の正規性を棄却する十分な根拠がないため、paired t-test を第一候補にしています。")
    else:
        st.info("差分の正規性を仮定しにくい、または判定困難なため、Wilcoxon signed-rank を第一候補にしています。")

    st.caption("※ 対応のある解析では、片方だけ欠損した行はペアとして扱えないため除外されます。")

    result_csv = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="検定結果をCSVでダウンロード",
        data=result_csv,
        file_name="paired_test_results.csv",
        mime="text/csv",
    )

else:
    st.subheader("反復測定ANOVA（1要因）")
    anova_table, pairwise_df, complete_df, long_df, anova_note = run_rm_anova(selected_df, alpha=alpha, correction_method=correction_method)

    st.write(f"完全データの対象数: **{len(complete_df)}**")

    fig_spaghetti = plot_spaghetti(complete_df)
    st.pyplot(fig_spaghetti)
    plt.close(fig_spaghetti)

    if anova_note:
        st.error(anova_note)
    else:
        st.markdown("**ANOVA結果**")
        st.dataframe(anova_table, use_container_width=True)

    st.markdown("**事後比較（対応のある t 検定 + 多重比較補正）**")
    st.dataframe(pairwise_df, use_container_width=True)
    st.info("この実装は 1要因の反復測定ANOVA です。事後比較は各条件ペアの対応のある t 検定に多重比較補正をかけています。")
    st.caption("※ 反復測定ANOVAでは、選択した全条件で完全データのある対象のみ解析に含めます。")

    anova_csv = anova_table.to_csv(index=False).encode("utf-8-sig") if anova_table is not None else b""
    pairwise_csv = pairwise_df.to_csv(index=False).encode("utf-8-sig")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="ANOVA結果をCSVでダウンロード",
            data=anova_csv,
            file_name="repeated_measures_anova_results.csv",
            mime="text/csv",
            disabled=anova_table is None,
        )
    with col_dl2:
        st.download_button(
            label="事後比較結果をCSVでダウンロード",
            data=pairwise_csv,
            file_name="posthoc_results.csv",
            mime="text/csv",
        )
