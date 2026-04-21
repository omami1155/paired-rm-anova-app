from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf


ページタイトル = "LMM（線形混合効果モデル）解析アプリ"
ページ説明 = (
    "wide形式のCSVを使って、群 × 条件 × 時間 の反復測定データを "
    "ランダム切片付きLMMで解析します。"
)
有意水準候補一覧 = [0.01, 0.05, 0.10]
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
class 解析設定:
    有意水準: float


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


def グラフ用条件表示を整える(値: object) -> str:
    return str(値).strip()


def グラフ用時間表示を整える(値: object) -> str:
    return str(値).strip()


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
        "このアプリは wide形式専用です。"
        " 1行=1サンプル、列=sample_id / group / condition / 各時点 の形で使ってください。"
    )


def サイドバーを表示する() -> 解析設定:
    st.sidebar.header("設定")
    有意水準 = st.sidebar.selectbox(
        "有意水準 α",
        options=有意水準候補一覧,
        index=1,
        format_func=lambda 値: f"{値:.2f}",
    )
    return 解析設定(有意水準=有意水準)


def サンプル説明を表示する() -> None:
    with st.expander("CSVの作り方", expanded=True):
        st.markdown(
            """
**このアプリで使うCSVは wide形式のみです。**

- 1行 = 1サンプル
- 必須の基本列 = `sample_id`, `group`, `condition`
- 時間列 = `t0`, `t1`, `t2` のように左から時系列順で並べます
- `condition` は加熱/非加熱に限らず、任意の条件名で使えます
- 測定値のセルには **数値だけ** を入れてください
- 欠測は空欄でかまいません

**おすすめの列順**

`sample_id, group, condition, t0, t1, t2, t3, t4, t5`

`sample_id` は **各 group × condition の中で 1〜10 を繰り返してOK** です。
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

    数式 = (
        f"value ~ C(group, levels={群水準一覧!r})"
        f" * C(condition, levels={条件水準一覧!r})"
        f" * C(time, levels={時間水準一覧!r})"
    )

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
    work["condition_display"] = work["condition"].map(グラフ用条件表示を整える)
    work["time_display"] = work["time"].astype(str).map(グラフ用時間表示を整える)
    summary = (
        work.groupby(["group", "condition_display", "time_display"], observed=True)["value"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    time_labels = [グラフ用時間表示を整える(x) for x in long_df["time"].cat.categories.tolist()]
    x_positions = np.arange(len(time_labels))

    for (group, condition), part in summary.groupby(["group", "condition_display"], observed=True):
        part = part.set_index("time_display").reindex(time_labels).reset_index()
        means = part["mean"].to_numpy(dtype=float)
        ax.plot(x_positions, means, marker="o", label=f"{group}-{condition}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Mean profiles by group and condition")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def create_spaghetti_subset_plot(long_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    selected_keys = long_df["subject_key"].drop_duplicates().tolist()[: min(30, long_df["subject_key"].nunique())]
    subset = long_df[long_df["subject_key"].isin(selected_keys)].copy()
    time_labels = [グラフ用時間表示を整える(x) for x in long_df["time"].cat.categories.tolist()]
    xmap = {label: idx for idx, label in enumerate(time_labels)}

    for _, part in subset.groupby("subject_key"):
        xs = [xmap[グラフ用時間表示を整える(x)] for x in part["time"].astype(str).tolist()]
        ax.plot(xs, part["value"].to_numpy(), marker="o", alpha=0.35)

    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Subject-level trajectories (first 30 samples)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


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
            "通常はまず `group:condition:time` の交互作用を確認し、"
            "必要に応じて別途追解析を検討してください。"
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


if __name__ == "__main__":
    メイン()
