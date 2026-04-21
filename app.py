from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ページタイトル = "LMM（線形混合効果モデル）解析アプリ"
ページ説明 = (
    "wide形式のCSVを使って、群 × 加熱 × 時間 の反復測定データを "
    "ランダム切片付きLMMで解析します。"
)
有意水準候補一覧 = [0.01, 0.05, 0.10]
補正法表示名 = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "FDR (Benjamini-Hochberg)",
}
既定時間列一覧 = ["直後", "1週", "2週", "3週", "4週", "5週"]
既定群一覧 = ["A", "B", "C", "D"]
既定加熱一覧 = ["加熱", "非加熱"]

見本CSV = """sample_id,group,heat,直後,1週,2週,3週,4週,5週
1,A,加熱,10.8,10.4,10.1,9.9,9.8,9.6
2,A,加熱,10.9,10.5,10.2,10.0,9.8,9.7
1,A,非加熱,10.7,10.6,10.5,10.4,10.4,10.3
2,A,非加熱,10.8,10.7,10.6,10.5,10.5,10.4
1,B,加熱,11.4,10.9,10.4,10.1,9.9,9.7
2,B,加熱,11.2,10.8,10.5,10.2,10.0,9.8
1,B,非加熱,11.3,11.1,10.9,10.8,10.7,10.6
2,B,非加熱,11.1,11.0,10.8,10.7,10.6,10.5
"""


@dataclass
class 解析設定:
    有意水準: float
    補正方法: str


@dataclass
class LMM適合結果:
    適合済みモデル: object | None
    数式: str
    適合法: str
    補足一覧: list[str]
    エラー内容: str
    全体検定表: pd.DataFrame | None
    係数表: pd.DataFrame | None


def csvをバイト列へ変換する(データフレーム: pd.DataFrame) -> bytes:
    return データフレーム.to_csv(index=False).encode("utf-8-sig")


def 安全に数値へ変換する(値) -> float:
    try:
        return float(値)
    except Exception:
        return np.nan


def 出現順で重複を除く(値一覧: pd.Series) -> list[str]:
    重複なし一覧: list[str] = []
    for 値 in 値一覧.astype(str).tolist():
        if 値 not in 重複なし一覧:
            重複なし一覧.append(値)
    return 重複なし一覧


def 加熱表示を整える(値: object) -> str:
    文字列 = str(値).strip().lower()
    if 文字列 in {"加熱", "heat", "heated"}:
        return "加熱"
    if 文字列 in {"非加熱", "no heat", "no_heat", "unheated", "non-heated", "non heated"}:
        return "非加熱"
    return str(値).strip()


def 時間表示を整える(値: object) -> str:
    文字列 = str(値).strip()
    小文字 = 文字列.lower()
    if 小文字 in {"直後", "immediate", "baseline", "0w", "t0"}:
        return "直後"
    if 文字列.endswith("週") and 文字列[:-1].isdigit():
        return 文字列
    if 小文字.endswith("w") and 小文字[:-1].isdigit():
        return f"{小文字[:-1]}週"
    return 文字列


def CSVを柔軟に読み込む(アップロードファイル) -> pd.DataFrame:
    生データ = アップロードファイル.getvalue()
    for 文字コード in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(生データ), encoding=文字コード)
        except Exception:
            continue
    raise ValueError("UTF-8 / UTF-8-SIG / CP932 のいずれでも読み込めませんでした。")


def 空のwideテンプレートを作る() -> bytes:
    行一覧: list[dict[str, object]] = []
    for 群名 in 既定群一覧:
        for 加熱名 in 既定加熱一覧:
            for サンプルID in range(1, 11):
                行データ: list[tuple[str, object]] = [
                    ("sample_id", サンプルID),
                    ("group", 群名),
                    ("heat", 加熱名),
                ]
                行データ.extend((時間列名, "") for 時間列名 in 既定時間列一覧)
                行一覧.append(dict(行データ))
    return pd.DataFrame(行一覧).to_csv(index=False).encode("utf-8-sig")


def ページを設定する() -> None:
    st.set_page_config(page_title=ページタイトル, layout="wide")
    st.title(ページタイトル)
    st.caption(ページ説明)
    st.info(
        "このアプリは wide形式専用です。"
        " 1行=1サンプル、列=sample_id / group / heat / 各時点 の形で使ってください。"
    )


def サイドバーを表示する() -> 解析設定:
    st.sidebar.header("設定")
    有意水準 = st.sidebar.selectbox(
        "有意水準 α",
        options=有意水準候補一覧,
        index=1,
        format_func=lambda 値: f"{値:.2f}",
    )
    補正方法 = st.sidebar.selectbox(
        "多重比較補正",
        options=list(補正法表示名),
        index=0,
        format_func=lambda キー: 補正法表示名[キー],
    )
    return 解析設定(有意水準=有意水準, 補正方法=補正方法)


def サンプル説明を表示する() -> None:
    with st.expander("CSVの作り方", expanded=True):
        st.markdown(
            """
**このアプリで使うCSVは wide形式のみです。**

- 1行 = 1サンプル
- 必須の基本列 = `sample_id`, `group`, `heat`
- 時間列 = `直後`, `1週`, `2週`, `3週`, `4週`, `5週` など
- 測定値のセルには **数値だけ** を入れてください
- 欠測は空欄でかまいません

**おすすめの列順**

`sample_id, group, heat, 直後, 1週, 2週, 3週, 4週, 5週`

`sample_id` は **各群 × 加熱条件の中で 1〜10 を繰り返してOK** です。
            """
        )

        st.subheader("見本CSV")
        st.code(見本CSV, language="csv")

        左列, 右列 = st.columns(2)
        with 左列:
            st.download_button(
                label="見本CSVをダウンロード",
                data=見本CSV.encode("utf-8-sig"),
                file_name="sample_lmm_wide.csv",
                mime="text/csv",
            )
        with 右列:
            st.download_button(
                label="80サンプルの空テンプレートをダウンロード",
                data=空のwideテンプレートを作る(),
                file_name="template_lmm_wide_80samples.csv",
                mime="text/csv",
            )


def 既定列を選ぶ(列名一覧: list[str], 候補列名一覧: list[str], 代替位置: int) -> str:
    for 候補列名 in 候補列名一覧:
        if 候補列名 in 列名一覧:
            return 候補列名
    return 列名一覧[min(代替位置, len(列名一覧) - 1)]


def wide形式を長形式へ変換する(元データ: pd.DataFrame) -> pd.DataFrame:
    st.subheader("列の確認")
    列名一覧 = 元データ.columns.tolist()

    既定サンプル列 = 既定列を選ぶ(列名一覧, ["sample_id", "id", "sample"], 0)
    既定群列 = 既定列を選ぶ(列名一覧, ["group", "群"], 1)
    既定加熱列 = 既定列を選ぶ(列名一覧, ["heat", "加熱"], 2)

    左列, 右列 = st.columns(2)
    with 左列:
        サンプルID列名 = st.selectbox("サンプルID列", options=列名一覧, index=列名一覧.index(既定サンプル列))
        群列名 = st.selectbox("群列", options=列名一覧, index=列名一覧.index(既定群列))
    with 右列:
        加熱列名 = st.selectbox("加熱列", options=列名一覧, index=列名一覧.index(既定加熱列))
        候補時間列一覧 = [列名 for 列名 in 列名一覧 if 列名 not in {サンプルID列名, 群列名, 加熱列名}]
        既定時間列候補 = [列名 for 列名 in 既定時間列一覧 if 列名 in 候補時間列一覧]
        if not 既定時間列候補:
            既定時間列候補 = 候補時間列一覧
        時間列一覧 = st.multiselect(
            "時間列（左から時系列順）",
            options=候補時間列一覧,
            default=既定時間列候補,
        )

    if len(時間列一覧) < 2:
        st.warning("時間列を2列以上選んでください。")
        return pd.DataFrame()

    作業データ = 元データ[[サンプルID列名, 群列名, 加熱列名] + 時間列一覧].copy()
    作業データ.columns = ["sample_id_raw", "group", "heat"] + 時間列一覧

    for 時間列名 in 時間列一覧:
        作業データ[時間列名] = pd.to_numeric(作業データ[時間列名], errors="coerce")

    長形式データ = 作業データ.melt(
        id_vars=["sample_id_raw", "group", "heat"],
        value_vars=時間列一覧,
        var_name="time",
        value_name="value",
    )
    長形式データ = 長形式データ.dropna(subset=["sample_id_raw", "group", "heat", "time", "value"]).copy()

    長形式データ["sample_id_raw"] = 長形式データ["sample_id_raw"].astype(str)
    長形式データ["group"] = 長形式データ["group"].astype(str)
    長形式データ["heat"] = 長形式データ["heat"].astype(str)
    長形式データ["time"] = pd.Categorical(
        長形式データ["time"].astype(str),
        categories=時間列一覧,
        ordered=True,
    )
    長形式データ["subject_key"] = (
        長形式データ["group"] + "|" + 長形式データ["heat"] + "|" + 長形式データ["sample_id_raw"]
    )
    長形式データ = 長形式データ.sort_values(
        ["group", "heat", "subject_key", "time"]
    ).reset_index(drop=True)
    return 長形式データ


def 長形式データを検証する(長形式データ: pd.DataFrame) -> tuple[list[str], list[str]]:
    警告一覧: list[str] = []
    エラー一覧: list[str] = []

    if 長形式データ.empty:
        エラー一覧.append("有効なデータがありません。")
        return 警告一覧, エラー一覧

    重複行あり = 長形式データ.duplicated(subset=["subject_key", "time"], keep=False)
    if 重複行あり.any():
        エラー一覧.append(
            "同じサンプル・同じ時点のデータが重複しています。"
            " 1サンプルにつき各時点1行だけになるよう確認してください。"
        )

    if 長形式データ["group"].nunique() < 2:
        エラー一覧.append("群が2水準未満です。")
    if 長形式データ["heat"].nunique() < 2:
        エラー一覧.append("加熱条件が2水準未満です。")
    if 長形式データ["time"].nunique() < 2:
        エラー一覧.append("時間が2水準未満です。")
    if 長形式データ["subject_key"].nunique() < 4:
        エラー一覧.append("サンプル数が少なすぎます。少なくとも4サンプル程度は必要です。")

    各サンプルの時点数 = 長形式データ.groupby("subject_key")["time"].nunique()
    総時点数 = 長形式データ["time"].nunique()
    if not 各サンプルの時点数.empty and 各サンプルの時点数.min() < 総時点数:
        警告一覧.append(
            "欠測が含まれています。LMMは実行できますが、一部の補助比較ではペア数が減ります。"
        )

    return 警告一覧, エラー一覧


def 表示用データを作る(長形式データ: pd.DataFrame) -> pd.DataFrame:
    表示用データ = 長形式データ.copy()
    表示用データ["heat"] = 表示用データ["heat"].map(加熱表示を整える)
    表示用データ["time"] = 表示用データ["time"].astype(str).map(時間表示を整える)
    表示用データ = 表示用データ.rename(
        columns={
            "sample_id_raw": "サンプルID",
            "group": "群",
            "heat": "加熱条件",
            "time": "時間",
            "value": "測定値",
            "subject_key": "サンプル識別キー",
        }
    )
    return 表示用データ[["サンプルID", "群", "加熱条件", "時間", "測定値", "サンプル識別キー"]]


def 記述統計を集計する(長形式データ: pd.DataFrame) -> pd.DataFrame:
    作業データ = 長形式データ.copy()
    作業データ["加熱表示"] = 作業データ["heat"].map(加熱表示を整える)
    作業データ["時間表示"] = 作業データ["time"].astype(str).map(時間表示を整える)

    記述統計表 = (
        作業データ.groupby(["group", "加熱表示", "時間表示"], observed=True)["value"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    記述統計表.columns = ["群", "加熱条件", "時間", "件数", "平均", "標準偏差", "中央値", "最小値", "最大値"]
    return 記述統計表


def LMMを適合する(長形式データ: pd.DataFrame, 有意水準: float) -> LMM適合結果:
    補足一覧: list[str] = []
    適合法 = ""
    エラー内容 = ""
    全体検定表 = None
    係数表 = None

    群水準一覧 = 出現順で重複を除く(長形式データ["group"])
    加熱水準一覧 = 出現順で重複を除く(長形式データ["heat"])
    時間水準一覧 = [str(値) for 値 in 長形式データ["time"].cat.categories.tolist()]

    数式 = (
        f"value ~ C(group, levels={群水準一覧!r})"
        f" * C(heat, levels={加熱水準一覧!r})"
        f" * C(time, levels={時間水準一覧!r})"
    )

    作業データ = 長形式データ.copy()
    作業データ["group"] = pd.Categorical(作業データ["group"], categories=群水準一覧)
    作業データ["heat"] = pd.Categorical(作業データ["heat"], categories=加熱水準一覧)
    作業データ["time"] = pd.Categorical(
        作業データ["time"].astype(str),
        categories=時間水準一覧,
        ordered=True,
    )

    適合済みモデル = None
    最後の例外 = None
    for 候補法 in ["lbfgs", "bfgs", "powell", "cg"]:
        try:
            model = smf.mixedlm(
                formula=数式,
                data=作業データ,
                groups=作業データ["subject_key"],
                re_formula="1",
            )
            適合済みモデル = model.fit(reml=False, method=候補法, maxiter=500, disp=False)
            適合法 = 候補法
            break
        except Exception as 例外:
            最後の例外 = 例外
            補足一覧.append(f"{候補法} 法では収束しませんでした: {例外}")

    if 適合済みモデル is None:
        エラー内容 = f"LMM の適合に失敗しました: {最後の例外}"
        return LMM適合結果(None, 数式, 適合法, 補足一覧, エラー内容, None, None)

    try:
        wald結果 = 適合済みモデル.wald_test_terms(scalar=True)
        全体検定表 = wald結果.table.reset_index().rename(
            columns={
                "index": "項",
                "statistic": "カイ二乗",
                "pvalue": "p値",
                "df_constraint": "自由度",
            }
        )
        全体検定表["判定"] = 全体検定表["p値"].apply(
            lambda p値: "有意" if pd.notna(p値) and p値 < 有意水準 else "有意差なし"
        )
    except Exception as 例外:
        補足一覧.append(f"固定効果の全体検定表を作れませんでした: {例外}")

    try:
        係数表 = pd.DataFrame(
            {
                "係数名": 適合済みモデル.fe_params.index,
                "推定値": 適合済みモデル.fe_params.values,
                "標準誤差": 適合済みモデル.bse_fe.values,
                "z値": 適合済みモデル.tvalues.loc[適合済みモデル.fe_params.index].values,
                "p値": 適合済みモデル.pvalues.loc[適合済みモデル.fe_params.index].values,
            }
        )
    except Exception as 例外:
        補足一覧.append(f"固定効果係数表を作れませんでした: {例外}")

    return LMM適合結果(
        適合済みモデル=適合済みモデル,
        数式=数式,
        適合法=適合法,
        補足一覧=補足一覧,
        エラー内容=エラー内容,
        全体検定表=全体検定表,
        係数表=係数表,
    )


def create_profile_plot(long_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    work = long_df.copy()
    work["heat_display"] = work["heat"].map(加熱表示を整える)
    work["time_display"] = work["time"].astype(str).map(時間表示を整える)
    summary = (
        work.groupby(["group", "heat_display", "time_display"], observed=True)["value"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    time_labels = [時間表示を整える(x) for x in long_df["time"].cat.categories.tolist()]
    x_positions = np.arange(len(time_labels))

    for (group, heat), part in summary.groupby(["group", "heat_display"], observed=True):
        part = part.set_index("time_display").reindex(time_labels).reset_index()
        means = part["mean"].to_numpy(dtype=float)
        counts = part["count"].fillna(0).to_numpy(dtype=float)
        sds = part["std"].fillna(0).to_numpy(dtype=float)
        sem = np.divide(sds, np.sqrt(np.maximum(counts, 1)), out=np.zeros_like(sds), where=np.maximum(counts, 1) > 0)
        ax.errorbar(x_positions, means, yerr=sem, marker="o", capsize=3, label=f"{group}-{heat}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Mean profiles by group and heat (error bars = SEM)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def create_spaghetti_subset_plot(long_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    selected_keys = long_df["subject_key"].drop_duplicates().tolist()[: min(30, long_df["subject_key"].nunique())]
    subset = long_df[long_df["subject_key"].isin(selected_keys)].copy()
    time_labels = [時間表示を整える(x) for x in long_df["time"].cat.categories.tolist()]
    xmap = {label: idx for idx, label in enumerate(time_labels)}

    for _, part in subset.groupby("subject_key"):
        xs = [xmap[時間表示を整える(x)] for x in part["time"].astype(str).tolist()]
        ax.plot(xs, part["value"].to_numpy(), marker="o", alpha=0.35)

    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Subject-level trajectories (first 30 samples)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def p値補正を適用する(
    結果行一覧: list[dict[str, object]],
    p値一覧: list[float],
    有意水準: float,
    補正方法: str,
) -> pd.DataFrame:
    有効な位置一覧 = [位置 for 位置, p値 in enumerate(p値一覧) if pd.notna(p値)]
    if 有効な位置一覧:
        有効p値一覧 = [p値一覧[位置] for 位置 in 有効な位置一覧]
        _, 補正後p値一覧, _, _ = multipletests(有効p値一覧, alpha=有意水準, method=補正方法)
        for 位置, 補正後p値 in zip(有効な位置一覧, 補正後p値一覧):
            結果行一覧[位置]["補正後p値"] = 安全に数値へ変換する(補正後p値)
            結果行一覧[位置]["判定"] = "有意" if 補正後p値 < 有意水準 else "有意差なし"

    for 結果行 in 結果行一覧:
        if 結果行.get("判定", "") == "":
            結果行["判定"] = "判定不可"
    return pd.DataFrame(結果行一覧)


def 加熱条件の事後比較を行う(長形式データ: pd.DataFrame, 有意水準: float, 補正方法: str) -> pd.DataFrame:
    結果行一覧: list[dict[str, object]] = []
    p値一覧: list[float] = []

    加熱水準一覧 = 出現順で重複を除く(長形式データ["heat"])
    if len(加熱水準一覧) != 2:
        return pd.DataFrame()

    比較対象A, 比較対象B = 加熱水準一覧[0], 加熱水準一覧[1]
    表示名A = 加熱表示を整える(比較対象A)
    表示名B = 加熱表示を整える(比較対象B)

    for (群名, 時点), 部分データ in 長形式データ.groupby(["group", "time"], observed=True):
        値A = 部分データ.loc[部分データ["heat"] == 比較対象A, "value"].dropna()
        値B = 部分データ.loc[部分データ["heat"] == 比較対象B, "value"].dropna()
        時点表示 = 時間表示を整える(時点)

        if len(値A) < 2 or len(値B) < 2:
            結果行一覧.append(
                {
                    "比較": f"{群名} / {時点表示}: {表示名A} 対 {表示名B}",
                    "n1": int(len(値A)),
                    "n2": int(len(値B)),
                    "平均差": np.nan,
                    "未補正p値": np.nan,
                    "補正後p値": np.nan,
                    "方法": "Welchのt検定",
                    "判定": "判定不可",
                }
            )
            p値一覧.append(np.nan)
            continue

        検定結果 = stats.ttest_ind(値A, 値B, equal_var=False, nan_policy="omit")
        p値 = 安全に数値へ変換する(検定結果.pvalue)
        結果行一覧.append(
            {
                "比較": f"{群名} / {時点表示}: {表示名A} 対 {表示名B}",
                "n1": int(len(値A)),
                "n2": int(len(値B)),
                "平均差": 安全に数値へ変換する(値A.mean() - 値B.mean()),
                "未補正p値": p値,
                "補正後p値": np.nan,
                "方法": "Welchのt検定",
                "判定": "",
            }
        )
        p値一覧.append(p値)

    return p値補正を適用する(結果行一覧, p値一覧, 有意水準, 補正方法)


def ベースライン比較を行う(長形式データ: pd.DataFrame, 有意水準: float, 補正方法: str) -> pd.DataFrame:
    結果行一覧: list[dict[str, object]] = []
    p値一覧: list[float] = []
    時間水準一覧 = [str(値) for 値 in 長形式データ["time"].cat.categories.tolist()]
    if len(時間水準一覧) < 2:
        return pd.DataFrame()

    ベースライン時点 = 時間水準一覧[0]
    ベースライン表示 = 時間表示を整える(ベースライン時点)

    for (群名, 加熱名), 部分データ in 長形式データ.groupby(["group", "heat"], observed=True):
        ペア比較表 = 部分データ.pivot_table(index="subject_key", columns="time", values="value", aggfunc="first")
        加熱表示 = 加熱表示を整える(加熱名)

        for 比較時点 in 時間水準一覧[1:]:
            比較時点表示 = 時間表示を整える(比較時点)
            if ベースライン時点 not in ペア比較表.columns or 比較時点 not in ペア比較表.columns:
                結果行一覧.append(
                    {
                        "比較": f"{群名} / {加熱表示}: {ベースライン表示} 対 {比較時点表示}",
                        "n": 0,
                        "平均差": np.nan,
                        "未補正p値": np.nan,
                        "補正後p値": np.nan,
                        "方法": "対応のあるt検定",
                        "判定": "判定不可",
                    }
                )
                p値一覧.append(np.nan)
                continue

            ペアデータ = ペア比較表[[ベースライン時点, 比較時点]].dropna()
            if len(ペアデータ) < 2:
                結果行一覧.append(
                    {
                        "比較": f"{群名} / {加熱表示}: {ベースライン表示} 対 {比較時点表示}",
                        "n": int(len(ペアデータ)),
                        "平均差": np.nan,
                        "未補正p値": np.nan,
                        "補正後p値": np.nan,
                        "方法": "対応のあるt検定",
                        "判定": "判定不可",
                    }
                )
                p値一覧.append(np.nan)
                continue

            検定結果 = stats.ttest_rel(ペアデータ[ベースライン時点], ペアデータ[比較時点], nan_policy="omit")
            p値 = 安全に数値へ変換する(検定結果.pvalue)
            結果行一覧.append(
                {
                    "比較": f"{群名} / {加熱表示}: {ベースライン表示} 対 {比較時点表示}",
                    "n": int(len(ペアデータ)),
                    "平均差": 安全に数値へ変換する(
                        ペアデータ[ベースライン時点].mean() - ペアデータ[比較時点].mean()
                    ),
                    "未補正p値": p値,
                    "補正後p値": np.nan,
                    "方法": "対応のあるt検定",
                    "判定": "",
                }
            )
            p値一覧.append(p値)

    return p値補正を適用する(結果行一覧, p値一覧, 有意水準, 補正方法)


def 解析データ概要を表示する(長形式データ: pd.DataFrame) -> None:
    st.subheader("解析に使う整形後データ")
    表示用整形データ = 表示用データを作る(長形式データ)
    st.dataframe(表示用整形データ.head(100), use_container_width=True)
    st.caption(
        f"行数: {len(表示用整形データ)} / サンプル数: {長形式データ['subject_key'].nunique()} / "
        f"群: {長形式データ['group'].nunique()} / 加熱: {長形式データ['heat'].nunique()} / 時間: {長形式データ['time'].nunique()}"
    )
    st.download_button(
        label="整形後データCSVをダウンロード",
        data=csvをバイト列へ変換する(表示用整形データ),
        file_name="prepared_long_data.csv",
        mime="text/csv",
    )


def 記述統計を表示する(長形式データ: pd.DataFrame) -> None:
    st.subheader("記述統計")
    記述統計表 = 記述統計を集計する(長形式データ)
    st.dataframe(記述統計表, use_container_width=True)
    st.download_button(
        label="記述統計CSVをダウンロード",
        data=csvをバイト列へ変換する(記述統計表),
        file_name="descriptive_statistics.csv",
        mime="text/csv",
    )

    左列, 右列 = st.columns(2)
    with 左列:
        fig = create_profile_plot(長形式データ)
        st.pyplot(fig)
        plt.close(fig)
    with 右列:
        fig = create_spaghetti_subset_plot(長形式データ)
        st.pyplot(fig)
        plt.close(fig)


def LMM結果を表示する(適合結果: LMM適合結果) -> None:
    st.subheader("LMM主解析")
    st.code(適合結果.数式)

    if 適合結果.エラー内容:
        st.error(適合結果.エラー内容)
        return

    st.success(f"LMM の適合に成功しました（適合法: {適合結果.適合法}）。")
    if 適合結果.補足一覧:
        for 補足 in 適合結果.補足一覧:
            st.warning(補足)

    if 適合結果.全体検定表 is not None:
        st.markdown("**固定効果の全体検定**")
        st.dataframe(適合結果.全体検定表, use_container_width=True)
        st.caption(
            "通常はまず `group:heat:time` の交互作用を見て、"
            "有意なら単純主効果や事後比較へ進みます。"
        )
        st.download_button(
            label="全体検定CSVをダウンロード",
            data=csvをバイト列へ変換する(適合結果.全体検定表),
            file_name="lmm_wald_terms.csv",
            mime="text/csv",
        )

    if 適合結果.係数表 is not None:
        st.markdown("**固定効果係数**")
        st.dataframe(適合結果.係数表, use_container_width=True)
        st.download_button(
            label="固定効果係数CSVをダウンロード",
            data=csvをバイト列へ変換する(適合結果.係数表),
            file_name="lmm_fixed_effects.csv",
            mime="text/csv",
        )


def 事後比較を表示する(長形式データ: pd.DataFrame, 設定: 解析設定) -> None:
    st.subheader("補助的な事後比較")
    st.caption(
        "ここは読みやすさを優先した補助解析です。主たる結論は上のLMM主解析を優先してください。"
    )

    加熱比較表 = 加熱条件の事後比較を行う(長形式データ, 設定.有意水準, 設定.補正方法)
    時間比較表 = ベースライン比較を行う(長形式データ, 設定.有意水準, 設定.補正方法)

    if not 加熱比較表.empty:
        st.markdown("**各群×各時点での加熱・非加熱比較**")
        st.dataframe(加熱比較表, use_container_width=True)
        st.download_button(
            label="加熱・非加熱比較CSVをダウンロード",
            data=csvをバイト列へ変換する(加熱比較表),
            file_name="posthoc_heat_within_group_time.csv",
            mime="text/csv",
        )

    if not 時間比較表.empty:
        st.markdown("**各群×加熱条件でのベースライン比較**")
        st.dataframe(時間比較表, use_container_width=True)
        st.download_button(
            label="ベースライン比較CSVをダウンロード",
            data=csvをバイト列へ変換する(時間比較表),
            file_name="posthoc_time_vs_baseline.csv",
            mime="text/csv",
        )


def メイン() -> None:
    ページを設定する()
    設定 = サイドバーを表示する()
    サンプル説明を表示する()

    アップロードファイル = st.file_uploader("CSVをアップロード", type=["csv"])
    if アップロードファイル is None:
        st.info("CSVをアップロードすると解析を開始します。")
        return

    try:
        元データ = CSVを柔軟に読み込む(アップロードファイル)
    except Exception as 例外:
        st.error(f"CSV の読み込みに失敗しました: {例外}")
        return

    st.subheader("データプレビュー")
    st.dataframe(元データ.head(50), use_container_width=True)

    長形式データ = wide形式を長形式へ変換する(元データ)
    if 長形式データ.empty:
        return

    警告一覧, エラー一覧 = 長形式データを検証する(長形式データ)
    for 警告 in 警告一覧:
        st.warning(警告)
    if エラー一覧:
        for エラー in エラー一覧:
            st.error(エラー)
        return

    解析データ概要を表示する(長形式データ)
    記述統計を表示する(長形式データ)

    適合結果 = LMMを適合する(長形式データ, 設定.有意水準)
    LMM結果を表示する(適合結果)

    if not 適合結果.エラー内容:
        事後比較を表示する(長形式データ, 設定)


if __name__ == "__main__":
    メイン()
