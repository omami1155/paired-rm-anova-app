from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy.contrasts import Sum
import streamlit as st
import statsmodels.formula.api as smf

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ページタイトル = "線形混合効果モデル解析"
ページ説明 = (
    "CSVを使って、群 × 条件 × 時間 の反復測定データを解析します。"
)
固定有意水準 = 0.05
既定時間列一覧 = ["t0", "t1", "t2", "t3", "t4", "t5"]
既定群一覧 = ["A", "B", "C", "D"]
既定条件一覧 = ["Condition_A", "Condition_B"]

見本CSV = """sample_id,group,condition,t0,t1,t2,t3,t4,t5
1,A,Condition_A,10.8,10.4,10.1,9.9,9.8,9.6
2,A,Condition_A,10.9,10.5,10.2,10.0,9.8,9.7
1,A,Condition_B,10.7,10.6,10.5,10.4,10.4,10.3
2,A,Condition_B,10.8,10.7,10.6,10.5,10.5,10.4
1,B,Condition_A,11.4,10.9,10.4,10.1,9.9,9.7
2,B,Condition_A,11.2,10.8,10.5,10.2,10.0,9.8
1,B,Condition_B,11.3,11.1,10.9,10.8,10.7,10.6
2,B,Condition_B,11.1,11.0,10.8,10.7,10.6,10.5
"""


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


def 検定項名を整える(項名: str) -> str:
    if 項名 == "Intercept":
        return "Intercept"

    整形後 = 項名
    for 元, 新 in (
        ("C(group, Sum)", "群"),
        ("C(condition, Sum)", "条件"),
        ("C(time, Sum)", "時間"),
        (":", "×"),
    ):
        整形後 = 整形後.replace(元, 新)
    return 整形後


def 係数名を整える(係数名: str) -> str:
    整形後 = 係数名
    for 元, 新 in (
        ("Intercept", "切片"),
        ("C(group, Sum)[S.", "群["),
        ("C(condition, Sum)[S.", "条件["),
        ("C(time, Sum)[S.", "時間["),
        (":", "×"),
    ):
        整形後 = 整形後.replace(元, 新)
    return 整形後


def 出現順で重複を除く(値一覧: pd.Series) -> list[str]:
    重複なし一覧: list[str] = []
    for 値 in 値一覧.astype(str).tolist():
        if 値 not in 重複なし一覧:
            重複なし一覧.append(値)
    return 重複なし一覧


def 条件表示を整える(値: object) -> str:
    return str(値).strip()


def 時間表示を整える(値: object) -> str:
    return str(値).strip()


def グラフ用時間対応表を作る(長形式データ: pd.DataFrame) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    元時間一覧 = [str(値) for 値 in 長形式データ["time"].cat.categories.tolist()]
    表示ラベル対応 = {元時間: f"T{番号}" for 番号, 元時間 in enumerate(元時間一覧, start=1)}
    対応表 = pd.DataFrame(
        {
            "Time code": list(表示ラベル対応.values()),
            "Original time": 元時間一覧,
        }
    )
    return 元時間一覧, 表示ラベル対応, 対応表


def グラフ用群対応表を作る(長形式データ: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    群一覧 = 出現順で重複を除く(長形式データ["group"])
    群コード対応 = {群名: f"G{番号}" for 番号, 群名 in enumerate(群一覧, start=1)}
    対応表 = pd.DataFrame(
        {
            "Group code": list(群コード対応.values()),
            "Original group": 群一覧,
        }
    )
    return 群コード対応, 対応表


def グラフ用条件対応表を作る(長形式データ: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    条件一覧 = 出現順で重複を除く(長形式データ["condition"])
    条件コード対応 = {条件名: f"C{番号}" for 番号, 条件名 in enumerate(条件一覧, start=1)}
    対応表 = pd.DataFrame(
        {
            "Condition code": list(条件コード対応.values()),
            "Original condition": 条件一覧,
        }
    )
    return 条件コード対応, 対応表


def グラフ用系列対応表を作る(長形式データ: pd.DataFrame) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    群コード対応, 群対応表 = グラフ用群対応表を作る(長形式データ)
    条件コード対応, 条件対応表 = グラフ用条件対応表を作る(長形式データ)
    群一覧 = 群対応表["Original group"].tolist()
    条件一覧 = 条件対応表["Original condition"].tolist()

    系列対応: dict[tuple[str, str], str] = {}
    対応表行一覧: list[dict[str, str]] = []
    for 群名 in 群一覧:
        for 条件名 in 条件一覧:
            系列コード = f"{群コード対応[群名]}-{条件コード対応[条件名]}"
            系列対応[(群名, 条件名)] = 系列コード
            対応表行一覧.append(
                {
                    "Series": 系列コード,
                    "Original group": 群名,
                    "Original condition": 条件名,
                }
            )

    return 系列対応, pd.DataFrame(対応表行一覧)


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
        for 条件名 in 既定条件一覧:
            for サンプルID in range(1, 11):
                行データ: list[tuple[str, object]] = [
                    ("sample_id", サンプルID),
                    ("group", 群名),
                    ("condition", 条件名),
                ]
                行データ.extend((時間列名, "") for 時間列名 in 既定時間列一覧)
                行一覧.append(dict(行データ))
    return pd.DataFrame(行一覧).to_csv(index=False).encode("utf-8-sig")


def ページを設定する() -> None:
    st.set_page_config(page_title=ページタイトル, layout="wide")
    st.title(ページタイトル)
    st.caption(ページ説明)
    st.info(
        " 1行=1サンプル、列=sample_id / group / condition / 各時点 の形で使ってください。"
    )


def サンプル説明を表示する() -> None:
    with st.expander("CSVの作り方", expanded=True):
        st.markdown(
            """
**このアプリで使うCSVは wide形式のみです。**

- 1行 = 1サンプル
- 必須の基本列 = `sample_id`, `group`, `condition`
- `sample_id` はサンプル番号、`group` は群名、`condition` は条件名です
- 時間列 = `t0`, `t1`, `t2` のように左から時系列順で並べます
- `condition` は加熱/非加熱に限らず、任意の条件名で使えます
- 測定値のセルには **数値だけ** を入れてください
- 欠測は空欄でかまいません

**おすすめの列順**

`sample_id, group, condition, t0, t1, t2, t3, t4, t5`
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
    既定条件列 = 既定列を選ぶ(列名一覧, ["condition", "heat", "条件", "加熱"], 2)

    左列, 右列 = st.columns(2)
    with 左列:
        サンプルID列名 = st.selectbox("サンプルID列", options=列名一覧, index=列名一覧.index(既定サンプル列))
        群列名 = st.selectbox("群列", options=列名一覧, index=列名一覧.index(既定群列))
    with 右列:
        条件列名 = st.selectbox("条件列", options=列名一覧, index=列名一覧.index(既定条件列))
        候補時間列一覧 = [列名 for 列名 in 列名一覧 if 列名 not in {サンプルID列名, 群列名, 条件列名}]
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

    作業データ = 元データ[[サンプルID列名, 群列名, 条件列名] + 時間列一覧].copy()
    作業データ.columns = ["sample_id_raw", "group", "condition"] + 時間列一覧

    for 時間列名 in 時間列一覧:
        作業データ[時間列名] = pd.to_numeric(作業データ[時間列名], errors="coerce")

    長形式データ = 作業データ.melt(
        id_vars=["sample_id_raw", "group", "condition"],
        value_vars=時間列一覧,
        var_name="time",
        value_name="value",
    )
    長形式データ = 長形式データ.dropna(subset=["sample_id_raw", "group", "condition", "time", "value"]).copy()

    長形式データ["sample_id_raw"] = 長形式データ["sample_id_raw"].astype(str)
    長形式データ["group"] = 長形式データ["group"].astype(str)
    長形式データ["condition"] = 長形式データ["condition"].astype(str)
    長形式データ["time"] = pd.Categorical(
        長形式データ["time"].astype(str),
        categories=時間列一覧,
        ordered=True,
    )
    長形式データ["subject_key"] = (
        長形式データ["group"] + "|" + 長形式データ["condition"] + "|" + 長形式データ["sample_id_raw"]
    )
    長形式データ = 長形式データ.sort_values(
        ["group", "condition", "subject_key", "time"]
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
    if 長形式データ["condition"].nunique() < 2:
        エラー一覧.append("条件が2水準未満です。")
    if 長形式データ["time"].nunique() < 2:
        エラー一覧.append("時間が2水準未満です。")
    if 長形式データ["subject_key"].nunique() < 4:
        エラー一覧.append("サンプル数が少なすぎます。少なくとも4サンプル程度は必要です。")

    各サンプルの時点数 = 長形式データ.groupby("subject_key")["time"].nunique()
    総時点数 = 長形式データ["time"].nunique()
    if not 各サンプルの時点数.empty and 各サンプルの時点数.min() < 総時点数:
        警告一覧.append(
            "欠測が含まれています。LMMは実行できますが、組み合わせによって推定精度が下がることがあります。"
        )

    return 警告一覧, エラー一覧


def 表示用データを作る(長形式データ: pd.DataFrame) -> pd.DataFrame:
    表示用データ = 長形式データ.copy()
    表示用データ["condition"] = 表示用データ["condition"].map(条件表示を整える)
    表示用データ["time"] = 表示用データ["time"].astype(str).map(時間表示を整える)
    表示用データ = 表示用データ.rename(
        columns={
            "sample_id_raw": "サンプルID",
            "group": "群",
            "condition": "条件",
            "time": "時間",
            "value": "測定値",
            "subject_key": "サンプル識別キー",
        }
    )
    return 表示用データ[["サンプルID", "群", "条件", "時間", "測定値", "サンプル識別キー"]]


def 記述統計を集計する(長形式データ: pd.DataFrame) -> pd.DataFrame:
    作業データ = 長形式データ.copy()
    作業データ["条件表示"] = 作業データ["condition"].map(条件表示を整える)
    作業データ["時間表示"] = 作業データ["time"].astype(str).map(時間表示を整える)

    記述統計表 = (
        作業データ.groupby(["group", "条件表示", "時間表示"], observed=True)["value"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    記述統計表.columns = ["群", "条件", "時間", "件数", "平均", "標準偏差", "中央値", "最小値", "最大値"]
    return 記述統計表


def LMMを適合する(長形式データ: pd.DataFrame, 有意水準: float) -> LMM適合結果:
    補足一覧: list[str] = []
    適合法 = ""
    エラー内容 = ""
    全体検定表 = None
    係数表 = None

    群水準一覧 = 出現順で重複を除く(長形式データ["group"])
    条件水準一覧 = 出現順で重複を除く(長形式データ["condition"])
    時間水準一覧 = [str(値) for 値 in 長形式データ["time"].cat.categories.tolist()]

    数式 = "value ~ C(group, Sum) * C(condition, Sum) * C(time, Sum)"

    作業データ = 長形式データ.copy()
    作業データ["group"] = pd.Categorical(作業データ["group"], categories=群水準一覧)
    作業データ["condition"] = pd.Categorical(作業データ["condition"], categories=条件水準一覧)
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
        全体検定表["項"] = 全体検定表["項"].map(検定項名を整える)
        全体検定表 = 全体検定表[全体検定表["項"] != "Intercept"].reset_index(drop=True)
        全体検定表["判定"] = 全体検定表["p値"].apply(
            lambda p値: "有意" if pd.notna(p値) and p値 < 有意水準 else "有意差なし"
        )
    except Exception as 例外:
        補足一覧.append(f"固定効果の全体検定表を作れませんでした: {例外}")

    try:
        係数表 = pd.DataFrame(
            {
                "係数名": [係数名を整える(係数名) for 係数名 in 適合済みモデル.fe_params.index],
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


def 水準順一覧を取得する(列: pd.Series) -> list[str]:
    if hasattr(列.dtype, "categories"):
        return [str(値) for 値 in 列.cat.categories.tolist()]
    return 出現順で重複を除く(列.astype(str))


def 効果図用ラベルを整える(値: object, ラベル対応: dict[str, str] | None = None) -> str:
    元ラベル = str(値)
    if ラベル対応 is None:
        return 元ラベル
    return ラベル対応.get(元ラベル, 元ラベル)


def 平均とCI95を集計する(データ: pd.DataFrame, グループ列: list[str]) -> pd.DataFrame:
    集計表 = (
        データ.groupby(グループ列, observed=True)["value"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    標準誤差 = 集計表["std"] / np.sqrt(集計表["count"].clip(lower=1))
    集計表["ci95"] = (1.96 * 標準誤差).fillna(0.0)
    return 集計表


def サンプル平均データを作る(長形式データ: pd.DataFrame) -> pd.DataFrame:
    return (
        長形式データ.groupby(["subject_key", "group", "condition"], observed=True)["value"]
        .mean()
        .reset_index()
    )


def 主効果プロットを作る(
    データ: pd.DataFrame,
    水準列: str,
    タイトル: str,
    x軸ラベル: str,
    点を重ねる: bool = True,
    ラベル対応: dict[str, str] | None = None,
):
    作図用 = データ.copy()
    作図用[水準列] = 作図用[水準列].astype(str)
    水準順一覧 = 水準順一覧を取得する(データ[水準列])
    集計表 = 平均とCI95を集計する(作図用, [水準列])
    集計表 = 集計表.set_index(水準列).reindex(水準順一覧).reset_index()

    x位置 = np.arange(len(水準順一覧))
    fig, ax = plt.subplots(figsize=(max(6.0, len(水準順一覧) * 1.35), 4.8))

    if 点を重ねる:
        for 番号, 水準 in enumerate(水準順一覧):
            値一覧 = np.sort(
                作図用.loc[作図用[水準列] == 水準, "value"].to_numpy(dtype=float)
            )
            if len(値一覧) == 0:
                continue
            if len(値一覧) == 1:
                x散布 = np.array([番号], dtype=float)
            else:
                x散布 = np.linspace(番号 - 0.12, 番号 + 0.12, len(値一覧))
            ax.scatter(x散布, 値一覧, color="#bdc3c7", s=28, alpha=0.7, edgecolors="none")

    ax.errorbar(
        x位置,
        集計表["mean"],
        yerr=集計表["ci95"],
        fmt="o-",
        color="#1f4e79",
        linewidth=2.2,
        markersize=7,
        capsize=4,
    )
    ax.set_xticks(x位置)
    ax.set_xticklabels([効果図用ラベルを整える(値, ラベル対応) for 値 in 水準順一覧])
    ax.set_xlabel(x軸ラベル)
    ax.set_ylabel("Value")
    ax.set_title(タイトル)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def 交互作用プロットを作る(
    データ: pd.DataFrame,
    x列: str,
    系列列: str,
    タイトル: str,
    x軸ラベル: str,
    凡例タイトル: str,
    xラベル対応: dict[str, str] | None = None,
    系列ラベル対応: dict[str, str] | None = None,
):
    作図用 = データ.copy()
    作図用[x列] = 作図用[x列].astype(str)
    作図用[系列列] = 作図用[系列列].astype(str)
    x順一覧 = 水準順一覧を取得する(データ[x列])
    系列順一覧 = 水準順一覧を取得する(データ[系列列])

    集計表 = 平均とCI95を集計する(作図用, [系列列, x列])
    x位置 = np.arange(len(x順一覧))
    色一覧 = plt.cm.tab10(np.linspace(0, 1, max(len(系列順一覧), 3)))

    fig, ax = plt.subplots(figsize=(max(6.8, len(x順一覧) * 1.4), 4.8))
    for 番号, 系列 in enumerate(系列順一覧):
        部分表 = 集計表[集計表[系列列] == 系列].set_index(x列).reindex(x順一覧).reset_index()
        ax.errorbar(
            x位置,
            部分表["mean"],
            yerr=部分表["ci95"],
            fmt="o-",
            linewidth=2.0,
            markersize=6,
            capsize=4,
            color=色一覧[番号 % len(色一覧)],
            label=効果図用ラベルを整える(系列, 系列ラベル対応),
        )

    ax.set_xticks(x位置)
    ax.set_xticklabels([効果図用ラベルを整える(値, xラベル対応) for 値 in x順一覧])
    ax.set_xlabel(x軸ラベル)
    ax.set_ylabel("Value")
    ax.set_title(タイトル)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=凡例タイトル)
    fig.tight_layout()
    return fig


def 三要因プロットを作る(長形式データ: pd.DataFrame, 系列ラベル対応: dict[str, str]):
    作図用 = 長形式データ.copy()
    作図用["系列"] = 作図用["group"].astype(str) + " / " + 作図用["condition"].astype(str)
    return 交互作用プロットを作る(
        作図用,
        x列="time",
        系列列="系列",
        タイトル="Group x condition trajectories",
        x軸ラベル="Time",
        凡例タイトル="Group / Condition",
        xラベル対応=グラフ用時間対応表を作る(長形式データ)[1],
        系列ラベル対応=系列ラベル対応,
    )


def 効果ごとの図を表示する(長形式データ: pd.DataFrame, 全体検定表: pd.DataFrame) -> None:
    表示順一覧 = ["群", "条件", "時間", "群×条件", "群×時間", "条件×時間", "群×条件×時間"]
    項一覧 = [項 for 項 in 表示順一覧 if 項 in 全体検定表["項"].tolist()]
    if not 項一覧:
        return

    サンプル平均表 = サンプル平均データを作る(長形式データ)
    検定結果辞書 = 全体検定表.set_index("項")[["p値", "判定"]].to_dict("index")
    群コード対応, 群対応表 = グラフ用群対応表を作る(長形式データ)
    条件コード対応, 条件対応表 = グラフ用条件対応表を作る(長形式データ)
    _時間一覧, 時間コード対応, 時間対応表 = グラフ用時間対応表を作る(長形式データ)
    系列コード対応, 系列対応表 = グラフ用系列対応表を作る(長形式データ)
    三要因系列ラベル対応 = {
        f"{群名} / {条件名}": 系列コード
        for (群名, 条件名), 系列コード in 系列コード対応.items()
    }

    st.markdown("**効果ごとの見取り図**")
    st.caption(
        "検定統計量ではなく、各効果でどの水準の値がどう違うかを見るための図です。"
        " 主効果の群・条件は各サンプルの時間平均、時間を含む効果は各時点の平均値を使っています。"
    )
    左列, 中央列, 右列 = st.columns(3)
    with 左列:
        st.dataframe(群対応表, use_container_width=True, hide_index=True)
    with 中央列:
        st.dataframe(条件対応表, use_container_width=True, hide_index=True)
    with 右列:
        st.dataframe(時間対応表, use_container_width=True, hide_index=True)
    with st.expander("Series code table"):
        st.dataframe(系列対応表, use_container_width=True, hide_index=True)

    タブ一覧 = st.tabs(項一覧)
    for 項, タブ in zip(項一覧, タブ一覧):
        with タブ:
            検定結果 = 検定結果辞書.get(項, {})
            p値 = 検定結果.get("p値")
            判定 = 検定結果.get("判定", "")
            if pd.notna(p値):
                st.caption(f"全体検定: p={p値:.3g} / 判定: {判定}")

            if 項 == "群":
                fig = 主効果プロットを作る(
                    サンプル平均表,
                    水準列="group",
                    タイトル="Mean value by group",
                    x軸ラベル="Group",
                    ラベル対応=群コード対応,
                )
                st.caption("各点は1サンプルの時間平均、線と誤差棒は平均値と95%CIです。")
            elif 項 == "条件":
                fig = 主効果プロットを作る(
                    サンプル平均表,
                    水準列="condition",
                    タイトル="Mean value by condition",
                    x軸ラベル="Condition",
                    ラベル対応=条件コード対応,
                )
                st.caption("各点は1サンプルの時間平均、線と誤差棒は平均値と95%CIです。")
            elif 項 == "時間":
                fig = 主効果プロットを作る(
                    長形式データ,
                    水準列="time",
                    タイトル="Mean value over time",
                    x軸ラベル="Time",
                    点を重ねる=False,
                    ラベル対応=時間コード対応,
                )
                st.caption("各点は各時点の平均値、誤差棒は95%CIです。")
            elif 項 == "群×条件":
                fig = 交互作用プロットを作る(
                    サンプル平均表,
                    x列="group",
                    系列列="condition",
                    タイトル="Group x condition means",
                    x軸ラベル="Group",
                    凡例タイトル="Condition",
                    xラベル対応=群コード対応,
                    系列ラベル対応=条件コード対応,
                )
                st.caption("各サンプルの時間平均を使っています。線が平行でないほど交互作用が示唆されます。")
            elif 項 == "群×時間":
                fig = 交互作用プロットを作る(
                    長形式データ,
                    x列="time",
                    系列列="group",
                    タイトル="Group trajectories over time",
                    x軸ラベル="Time",
                    凡例タイトル="Group",
                    xラベル対応=時間コード対応,
                    系列ラベル対応=群コード対応,
                )
                st.caption("群ごとの時間推移差を見る図です。線の形が違うほど交互作用が示唆されます。")
            elif 項 == "条件×時間":
                fig = 交互作用プロットを作る(
                    長形式データ,
                    x列="time",
                    系列列="condition",
                    タイトル="Condition trajectories over time",
                    x軸ラベル="Time",
                    凡例タイトル="Condition",
                    xラベル対応=時間コード対応,
                    系列ラベル対応=条件コード対応,
                )
                st.caption("条件ごとの時間推移差を見る図です。線の形が違うほど交互作用が示唆されます。")
            else:
                fig = 三要因プロットを作る(長形式データ, 三要因系列ラベル対応)
                st.caption("群と条件の組み合わせごとの時間推移を並べて、三要因の違いを見やすくしています。")

            st.pyplot(fig)
            plt.close(fig)


def 解析データ概要を表示する(長形式データ: pd.DataFrame) -> None:
    st.subheader("解析に使う整形後データ")
    表示用整形データ = 表示用データを作る(長形式データ)
    st.dataframe(表示用整形データ.head(100), use_container_width=True)
    st.caption(
        f"行数: {len(表示用整形データ)} / サンプル数: {長形式データ['subject_key'].nunique()} / "
        f"群: {長形式データ['group'].nunique()} / 条件: {長形式データ['condition'].nunique()} / 時間: {長形式データ['time'].nunique()}"
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


def LMM結果を表示する(適合結果: LMM適合結果, 長形式データ: pd.DataFrame) -> None:
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
            "偏差コントラストによる Wald のカイ二乗検定です。"
            " 参照水準やCSVの行順に依存しにくい形で、切片は表示から外しています。"
            f" 判定基準は α={固定有意水準:.2f} 固定です。"
        )
        効果ごとの図を表示する(長形式データ, 適合結果.全体検定表)
        st.download_button(
            label="全体検定CSVをダウンロード",
            data=csvをバイト列へ変換する(適合結果.全体検定表),
            file_name="lmm_wald_terms.csv",
            mime="text/csv",
        )

    if 適合結果.係数表 is not None:
        st.markdown("**固定効果係数**")
        st.dataframe(適合結果.係数表, use_container_width=True)
        st.caption("係数は偏差コントラストで推定されるため、各水準の偏差として解釈してください。")
        st.download_button(
            label="固定効果係数CSVをダウンロード",
            data=csvをバイト列へ変換する(適合結果.係数表),
            file_name="lmm_fixed_effects.csv",
            mime="text/csv",
        )


def メイン() -> None:
    ページを設定する()
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

    適合結果 = LMMを適合する(長形式データ, 固定有意水準)
    LMM結果を表示する(適合結果, 長形式データ)


if __name__ == "__main__":
    メイン()
