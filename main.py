# main.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt


def set_matplotlib_dark_theme():
    """Tema consistente para matplotlib no Streamlit dark."""
    plt.rcParams.update({
        "figure.facecolor": (0, 0, 0, 0),
        "axes.facecolor": (0, 0, 0, 0),
        "savefig.facecolor": (0, 0, 0, 0),
        "text.color": "#E8EEF6",
        "axes.labelcolor": "#E8EEF6",
        "axes.edgecolor": (232/255, 238/255, 246/255, 0.25),
        "xtick.color": (232/255, 238/255, 246/255, 0.70),
        "ytick.color": (232/255, 238/255, 246/255, 0.70),
        "grid.color": (232/255, 238/255, 246/255, 0.12),
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Painel ML — Obesidade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_obesity_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
DATA_PATH = BASE_DIR / "obesity_dataset_modelo.csv"


# ---------------------------
# Modern minimal UI (CSS)
# ---------------------------
def inject_css():
    st.markdown(
        """
<style>
:root{
  --radius: 18px;
  --shadow: 0 10px 30px rgba(0,0,0,0.28);
  --shadow2: 0 8px 18px rgba(0,0,0,0.20);

  --text: rgba(232, 238, 246, 0.96);
  --muted: rgba(232, 238, 246, 0.64);
  --muted2: rgba(232, 238, 246, 0.46);
  --border: rgba(232, 238, 246, 0.10);

  --card: rgba(255,255,255,0.05);
  --card2: rgba(255,255,255,0.035);
  --hero: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
}

/* Reduce padding */
.block-container{
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.25rem !important;
}

/* Hero */
.hero{
  border: 1px solid var(--border);
  background: var(--hero);
  border-radius: var(--radius);
  padding: 18px 18px;
  box-shadow: var(--shadow);
}
.hero .title{
  font-size: 34px;
  font-weight: 750;
  line-height: 1.15;
  color: var(--text);
  margin: 0 0 6px 0;
}
.hero .subtitle{
  color: var(--muted);
  font-size: 14px;
  margin: 0;
}
.badges{
  display:flex; gap:8px; flex-wrap:wrap;
  margin-top: 14px;
}
.badge{
  display:inline-flex; align-items:center; gap:8px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
}

/* Section titles */
.section-title{
  margin: 0 0 12px 0;
  font-size: 13px;
  letter-spacing: 0.22em;
  color: var(--muted);
  text-transform: uppercase;
}

/* ✅ REAL cards para widgets: container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow2) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 14px 14px !important;
}

/* Subcard */
.subcard{
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  border-radius: var(--radius);
  padding: 12px 12px;
}

/* KPI */
.kpi-row{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}
.kpi{
  border: 1px solid var(--border);
  background: var(--card2);
  border-radius: 16px;
  padding: 12px 12px;
  min-height: 118px;
}
.kpi .label{
  color: var(--muted);
  font-size: 12px;
  margin: 0 0 6px 0;
}
.kpi .value{
  color: var(--text);
  font-size: 20px;
  font-weight: 780;
  margin: 0;
  line-height: 1.22;

  /* ✅ evita quebrar letra por letra, mas permite quebrar por espaços */
  word-break: normal;
  overflow-wrap: break-word;
  white-space: normal;
}
.kpi .value.small{
  font-size: 18px;
}
.kpi .hint{
  color: var(--muted2);
  font-size: 12px;
  margin: 6px 0 0 0;
}

/* Dataframe */
div[data-testid="stDataFrame"]{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

/* Hide Streamlit default footer/menu */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
</style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ---------------------------
# Helpers / Model metadata
# ---------------------------
CLASS_NAME = {
    0: "Abaixo do peso",
    1: "Peso normal",
    5: "Sobrepeso (Nível I)",
    6: "Sobrepeso (Nível II)",
    2: "Obesidade (Tipo I)",
    3: "Obesidade (Tipo II)",
    4: "Obesidade (Tipo III)",
}


def safe_load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.name} (esperado em: {path})")
    return joblib.load(path)


@st.cache_resource
def load_model_and_features():
    model = safe_load_joblib(MODEL_PATH)
    features = safe_load_joblib(FEATURES_PATH)
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError("model_features.pkl precisa ser uma LISTA de strings (colunas do X).")
    return model, features


@st.cache_data
def load_data_for_eda():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def label_class(v) -> str:
    try:
        return CLASS_NAME.get(int(v), str(v))
    except Exception:
        return str(v)


def build_feature_row(
    features_order: list[str],
    age: float,
    height: float,
    weight: float,
    fcvc: int,
    ncp: int,
    ch2o: int,
    faf: int,
    tue: int,
    gender: str,
    family_history: str,
    favc: str,
    caec: str,
    smoke: str,
    scc: str,
    calc: str,
    mtrans: str,
    bmi_override: float | None,
) -> pd.DataFrame:
    row = {col: 0.0 for col in features_order}

    # Numeric
    row["Age"] = float(age)
    row["Height"] = float(height)
    row["Weight"] = float(weight)
    row["FCVC"] = float(fcvc)
    row["NCP"] = float(ncp)
    row["CH2O"] = float(ch2o)
    row["FAF"] = float(faf)
    row["TUE"] = float(tue)

    bmi_calc = float(weight) / (float(height) ** 2) if height > 0 else 0.0
    row["BMI"] = float(bmi_override) if (bmi_override is not None) else bmi_calc

    # Binary
    if gender == "Male":
        row["Gender_Male"] = 1.0
    if family_history == "Sim":
        row["family_history_yes"] = 1.0
    if favc == "Sim":
        row["FAVC_yes"] = 1.0
    if smoke == "Sim":
        row["SMOKE_yes"] = 1.0
    if scc == "Sim":
        row["SCC_yes"] = 1.0

    # One-hot CAEC (base Always)
    if caec == "No":
        row["CAEC_no"] = 1.0
    elif caec == "Sometimes":
        row["CAEC_Sometimes"] = 1.0
    elif caec == "Frequently":
        row["CAEC_Frequently"] = 1.0

    # One-hot CALC (base Always)
    if calc == "No":
        row["CALC_no"] = 1.0
    elif calc == "Sometimes":
        row["CALC_Sometimes"] = 1.0
    elif calc == "Frequently":
        row["CALC_Frequently"] = 1.0

    # One-hot MTRANS (base Automobile)
    if mtrans == "Walking":
        row["MTRANS_Walking"] = 1.0
    elif mtrans == "Public Transportation":
        row["MTRANS_Public_Transportation"] = 1.0
    elif mtrans == "Bike":
        row["MTRANS_Bike"] = 1.0
    elif mtrans == "Motorbike":
        row["MTRANS_Motorbike"] = 1.0

    return pd.DataFrame([row], columns=features_order).astype(float)


def probs_table(model, X_row: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Seu modelo não tem predict_proba().")
    proba = model.predict_proba(X_row)[0]
    classes = list(model.classes_)
    dfp = pd.DataFrame({"Classe": classes, "Probabilidade": proba})
    dfp["Rótulo"] = dfp["Classe"].map(CLASS_NAME).fillna(dfp["Classe"].astype(str))
    dfp = dfp.sort_values("Probabilidade", ascending=False).reset_index(drop=True)
    return dfp


def fig_base(title: str):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title(title, fontsize=12, pad=10)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    return fig, ax


# ---------------------------
# Header (hero)
# ---------------------------
st.markdown(
    """
<div class="hero">
  <div class="title">📊 Painel ML — Obesidade</div>
  <p class="subtitle">Predição em tempo real e dashboard exploratório.</p>
  <div class="badges">
    <div class="badge">🧠 Modelo: Random Forest</div>
    <div class="badge">⚡ Atualização automática</div>
    <div class="badge">📁 Fonte: CSV + Features (.pkl)</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------
# Load model
# ---------------------------
try:
    model, features_order = load_model_and_features()
except ModuleNotFoundError as e:
    st.error(f"Faltou instalar uma dependência do modelo: {e}. Rode: python -m pip install scikit-learn")
    st.stop()
except Exception as e:
    st.error(
        "Não consegui carregar o modelo/feature list. "
        "Confere se `random_forest_obesity_model.pkl` e `model_features.pkl` estão na mesma pasta do main.py.\n\n"
        f"Detalhe: {e}"
    )
    st.stop()


# ---------------------------
# Sidebar (executive)
# ---------------------------
st.sidebar.markdown("## Navegação")
menu = st.sidebar.radio(" ", ["🔮 Previsão", "📈 Análise Exploratória"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## Filtros (EDA)")
df_eda = load_data_for_eda()

if df_eda is None:
    st.sidebar.caption("⚠️ `obesity_dataset_modelo.csv` não encontrado (EDA indisponível).")
else:
    class_options = sorted(df_eda["Obesity"].unique().tolist())
    class_labels = {c: CLASS_NAME.get(int(c), str(c)) for c in class_options}

    with st.sidebar.expander("Classes (Obesity)", expanded=True):
        selected_classes = []
        for c in class_options:
            if st.checkbox(class_labels[c], value=True, key=f"class_{c}"):
                selected_classes.append(c)

    gender_filter = st.sidebar.radio("Gênero", ["Todos", "Female", "Male"], index=0)

    age_min, age_max = int(df_eda["Age"].min()), int(df_eda["Age"].max())
    age_range = st.sidebar.slider("Faixa Etária", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    consumo_agua = st.sidebar.radio("Consumo de Água (CH2O)", ["Todos", "Baixo (1)", "Médio (2)", "Alto (3)"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Arquivos na pasta do app: random_forest_obesity_model.pkl • model_features.pkl • obesity_dataset_modelo.csv")


# ===========================
# PAGE: PREDICTION
# ===========================
if menu == "🔮 Previsão":
    left, right = st.columns([1.05, 1.25], gap="large")

    with left:
        st.markdown('<div class="section-title">Entradas</div>', unsafe_allow_html=True)

        # ✅ Card real (não cria div vazio)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Idade", 10, 80, 25)
                height = st.number_input("Altura (m)", min_value=1.0, max_value=2.3, value=1.70, step=0.01, format="%.2f")
            with c2:
                weight = st.number_input("Peso (kg)", min_value=25.0, max_value=250.0, value=75.0, step=0.5)

            st.markdown("#### Hábitos")
            fcvc = st.slider("Vegetais (FCVC)", 1, 3, 2)
            ncp = st.slider("Nº refeições (NCP)", 1, 4, 3)
            ch2o = st.slider("Água (CH2O)", 1, 3, 2)
            faf = st.slider("Atividade física (FAF)", 0, 3, 1)
            tue = st.slider("Tempo em telas (TUE)", 0, 2, 1)

            st.markdown("#### Perfil")
            gender = st.radio("Gênero", ["Female", "Male"], horizontal=True)
            family_history = st.radio("Histórico familiar de sobrepeso?", ["Não", "Sim"], horizontal=True)
            favc = st.radio("Alta caloria frequente (FAVC)?", ["Não", "Sim"], horizontal=True)

            c3, c4 = st.columns(2)
            with c3:
                caec = st.selectbox("Beliscos (CAEC)", ["Always", "Sometimes", "Frequently", "No"], index=1)
                smoke = st.radio("Fuma?", ["Não", "Sim"], horizontal=True)
            with c4:
                scc = st.radio("Monitora calorias (SCC)?", ["Não", "Sim"], horizontal=True)
                calc = st.selectbox("Álcool (CALC)", ["Always", "Sometimes", "Frequently", "No"], index=1)

            mtrans = st.selectbox(
                "Transporte (MTRANS)",
                ["Automobile", "Public Transportation", "Walking", "Bike", "Motorbike"],
                index=0,
            )

            st.markdown("#### IMC")
            bmi_auto = (weight / (height ** 2)) if height > 0 else 0.0
            st.caption(f"IMC calculado: **{bmi_auto:.2f}**")

            use_override = st.checkbox("Editar IMC manualmente", value=False)
            bmi_override = None
            if use_override:
                bmi_override = st.number_input("IMC (override)", min_value=5.0, max_value=80.0, value=float(bmi_auto), step=0.1)

        with st.container(border=True):
            st.markdown("#### Notas")
            st.markdown(
                '<div class="small">A previsão atualiza automaticamente conforme você altera os campos. '
                'Os valores e categorias dependem da engenharia de features usada no treino.</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">Resultado do modelo</div>', unsafe_allow_html=True)

        try:
            X_row = build_feature_row(
                features_order=features_order,
                age=age,
                height=height,
                weight=weight,
                fcvc=fcvc,
                ncp=ncp,
                ch2o=ch2o,
                faf=faf,
                tue=tue,
                gender=gender,
                family_history=family_history,
                favc=favc,
                caec=caec,
                smoke=smoke,
                scc=scc,
                calc=calc,
                mtrans=mtrans,
                bmi_override=bmi_override,
            )

            dfp = probs_table(model, X_row)
            top = dfp.iloc[0]

            # ✅ Card real (não cria div vazio)
            with st.container(border=True):
                # classe prevista pode ser longa: usa .value.small
                classe_html = f"""
<div class="kpi-row">
  <div class="kpi">
    <p class="label">Classe prevista</p>
    <p class="value small">{top['Rótulo']}</p>
    <p class="hint">Top-1</p>
  </div>
  <div class="kpi">
    <p class="label">Probabilidade</p>
    <p class="value">{top['Probabilidade']*100:.1f}%</p>
    <p class="hint">Top-1</p>
  </div>
  <div class="kpi">
    <p class="label">IMC</p>
    <p class="value">{(bmi_override if bmi_override is not None else bmi_auto):.2f}</p>
    <p class="hint">calculado/override</p>
  </div>
  <div class="kpi">
    <p class="label">Idade</p>
    <p class="value">{int(age)}</p>
    <p class="hint">anos</p>
  </div>
  <div class="kpi">
    <p class="label">Água (CH2O)</p>
    <p class="value">{int(ch2o)}</p>
    <p class="hint">escala 1–3</p>
  </div>
</div>
"""
                st.markdown(classe_html, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("#### Probabilidade por classe")
                st.dataframe(
                    dfp[["Rótulo", "Probabilidade"]].style.format({"Probabilidade": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown('<div class="small">Ordenado por maior probabilidade.</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("#### Visualização")
                fig, ax = fig_base("Distribuição das probabilidades")
                ax.bar(dfp["Rótulo"], dfp["Probabilidade"], alpha=0.9)
                ax.set_ylabel("Probabilidade")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig, clear_figure=True)

        except Exception as e:
            st.error(f"Erro ao prever: {e}")


# ===========================
# PAGE: EDA DASHBOARD
# ===========================
elif menu == "📈 Análise Exploratória":
    set_matplotlib_dark_theme()
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)

    if df_eda is None:
        st.warning("Para ver a EDA, coloque `obesity_dataset_modelo.csv` na mesma pasta do `main.py`.")
        st.stop()

    df_f = df_eda.copy()

    if "selected_classes" in locals() and len(selected_classes) > 0:
        df_f = df_f[df_f["Obesity"].isin(selected_classes)]

    if "gender_filter" in locals() and gender_filter != "Todos":
        df_f = df_f[df_f["Gender_Male"] == (1 if gender_filter == "Male" else 0)]

    if "age_range" in locals():
        df_f = df_f[(df_f["Age"] >= age_range[0]) & (df_f["Age"] <= age_range[1])]

    if "consumo_agua" in locals() and consumo_agua != "Todos":
        target = int(consumo_agua.split("(")[1].split(")")[0])
        df_f = df_f[df_f["CH2O"] == target]

    if df_f.empty:
        st.warning("Nenhum dado após os filtros. Ajuste os filtros na sidebar.")
        st.stop()

    # KPIs
    n = len(df_f)
    bmi_mean = df_f["BMI"].mean()
    age_mean = df_f["Age"].mean()
    water_mean = df_f["CH2O"].mean()
    dist_class = df_f["Obesity"].value_counts().sort_index()
    top_class = dist_class.idxmax()
    top_class_name = label_class(top_class)
    top_share = dist_class.max() / n if n else 0

    with st.container(border=True):
        st.markdown(
            f"""
<div class="kpi-row">
  <div class="kpi">
    <p class="label">Indivíduos</p>
    <p class="value">{f"{n:,}".replace(",", ".")}</p>
    <p class="hint">após filtros</p>
  </div>
  <div class="kpi">
    <p class="label">Classe dominante</p>
    <p class="value small">{top_class_name}</p>
    <p class="hint">{top_share*100:.1f}%</p>
  </div>
  <div class="kpi">
    <p class="label">IMC médio</p>
    <p class="value">{bmi_mean:.2f}</p>
    <p class="hint">média</p>
  </div>
  <div class="kpi">
    <p class="label">Idade média</p>
    <p class="value">{age_mean:.1f}</p>
    <p class="hint">anos</p>
  </div>
  <div class="kpi">
    <p class="label">Água média (CH2O)</p>
    <p class="value">{water_mean:.2f}</p>
    <p class="hint">escala 1–3</p>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    def fig_base(title: str):
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title(title, fontsize=12, pad=10)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        return fig, ax

    def bar_plot(x, y, title, ylabel=""):
        fig, ax = fig_base(title)
        ax.bar(x, y, alpha=0.9)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(ylabel)
        st.pyplot(fig, clear_figure=True)

    def hist_plot(series, title, xlabel):
        fig, ax = fig_base(title)
        ax.hist(series.dropna().values, bins=20, alpha=0.9, edgecolor=(1,1,1,0.12))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    def boxplot_by_class(df, col, title):
        classes = sorted(df["Obesity"].unique().tolist())
        data = [df[df["Obesity"] == c][col].dropna().values for c in classes]
        labels = [label_class(c) for c in classes]
        fig, ax = fig_base(title)

        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set_alpha(0.25)
            box.set_edgecolor((1,1,1,0.25))
        for median in bp["medians"]:
            median.set_color("#E8EEF6")
            median.set_linewidth(1.2)
        for w in bp["whiskers"]:
            w.set_color((1,1,1,0.18))
        for c in bp["caps"]:
            c.set_color((1,1,1,0.18))

        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(col)
        st.pyplot(fig, clear_figure=True)

    def correlation_heatmap(df, cols, title):
        corr = df[cols].corr(numeric_only=True)
        fig, ax = fig_base(title)
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#E8EEF6")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)

    def scatter(df, x, y, title):
        fig, ax = fig_base(title)
        ax.scatter(df[x], df[y], s=12, alpha=0.7, c=df["Obesity"])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        st.pyplot(fig, clear_figure=True)

    st.markdown('<div class="section-title">Composição</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")

    with r1c1:
        with st.container(border=True):
            st.markdown("#### Distribuição por classe")
            vc = df_f["Obesity"].value_counts().sort_values(ascending=False)
            bar_plot([label_class(i) for i in vc.index], vc.values, " ", ylabel="Count")

    with r1c2:
        with st.container(border=True):
            st.markdown("#### Distribuição por gênero")
            gender_counts = pd.Series(
                {"Female": int((df_f["Gender_Male"] == 0).sum()), "Male": int((df_f["Gender_Male"] == 1).sum())}
            )
            bar_plot(gender_counts.index.tolist(), gender_counts.values, " ", ylabel="Count")

    with r1c3:
        with st.container(border=True):
            st.markdown("#### Transporte (estimado)")
            mtrans_cols = [c for c in df_f.columns if c.startswith("MTRANS_")]
            if mtrans_cols:
                tmp = df_f[mtrans_cols].copy()
                tmp["Automobile"] = (tmp.sum(axis=1) == 0).astype(int)
                m_counts = tmp.sum(axis=0).sort_values(ascending=False)
                labels = [s.replace("MTRANS_", "").replace("_", " ") for s in m_counts.index.tolist()]
                bar_plot(labels, m_counts.values, " ", ylabel="Count")
            else:
                st.info("Colunas MTRANS_* não encontradas no dataset para esta visualização.")

    st.markdown('<div class="section-title">Distribuições</div>', unsafe_allow_html=True)
    r2a, r2b, r2c = st.columns(3, gap="large")
    with r2a:
        with st.container(border=True):
            st.markdown("#### IMC")
            hist_plot(df_f["BMI"], " ", "IMC")
    with r2b:
        with st.container(border=True):
            st.markdown("#### Idade")
            hist_plot(df_f["Age"], " ", "Idade")
    with r2c:
        with st.container(border=True):
            st.markdown("#### Água (CH2O)")
            hist_plot(df_f["CH2O"], " ", "CH2O")

    r2d, r2e = st.columns(2, gap="large")
    with r2d:
        with st.container(border=True):
            st.markdown("#### IMC por classe")
            boxplot_by_class(df_f, "BMI", " ")
    with r2e:
        with st.container(border=True):
            st.markdown("#### Idade por classe")
            boxplot_by_class(df_f, "Age", " ")

    st.markdown('<div class="section-title">Relações</div>', unsafe_allow_html=True)
    numeric_cols = [c for c in ["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"] if c in df_f.columns]
    r3a, r3b = st.columns(2, gap="large")
    with r3a:
        with st.container(border=True):
            st.markdown("#### Correlação")
            correlation_heatmap(df_f, numeric_cols, " ")
    with r3b:
        with st.container(border=True):
            st.markdown("#### Idade × IMC")
            scatter(df_f, "Age", "BMI", " ")

    r3c, r3d = st.columns(2, gap="large")
    with r3c:
        with st.container(border=True):
            st.markdown("#### Altura × Peso")
            scatter(df_f, "Height", "Weight", " ")
    with r3d:
        with st.container(border=True):
            st.markdown("#### Atividade (FAF) × IMC")
            scatter(df_f, "FAF", "BMI", " ")

    st.markdown('<div class="section-title">Tabelas</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Resumo por classe")
        group = (
            df_f.groupby("Obesity", as_index=False)
            .agg(
                n=("Obesity", "size"),
                bmi_mean=("BMI", "mean"),
                age_mean=("Age", "mean"),
                water_mean=("CH2O", "mean"),
                activity_mean=("FAF", "mean"),
            )
        )
        group["Classe"] = group["Obesity"].apply(label_class)
        group["%"] = group["n"] / group["n"].sum()
        group = group[["Classe", "n", "%", "bmi_mean", "age_mean", "water_mean", "activity_mean"]].sort_values("n", ascending=False)

        st.dataframe(
            group.style.format(
                {
                    "%": "{:.1%}",
                    "bmi_mean": "{:.2f}",
                    "age_mean": "{:.1f}",
                    "water_mean": "{:.2f}",
                    "activity_mean": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('<div class="small">Use os filtros na sidebar para fatiar o dashboard.</div>', unsafe_allow_html=True)

    with st.expander("Ver amostra dos dados filtrados"):
        st.dataframe(df_f.head(50), use_container_width=True, hide_index=True)
