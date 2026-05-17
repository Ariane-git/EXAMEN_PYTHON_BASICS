import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, io, sys, warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Analyse RH - Churn", layout="wide")

# ─── Style CSS Épuré & Moderne (Thème Lumineux) ──────────────────────────────
# ─── Style CSS Épuré & Moderne (Thème Bleu Nuit / Dark Mode) ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fond principal de l'application (Bleu Nuit) */
.stApp {
    background-color: #0b1329;
}

/* Sidebar moderne (Légèrement plus sombre pour détacher du fond) */
[data-testid="stSidebar"] {
    background-color: #080c18 !important;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.05);
}

/* Titres en Blanc / Bleu très clair */
h1 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    margin-bottom: 0.5rem !important;
}
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* Cartes Métriques Bleu Nuit Intermédiaire (Effet Glassmorphism) */
[data-testid="metric-container"] {
    background-color: #1c2541 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; /* Valeur en bleu flashy pour bien ressortir */
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Séparateur discret */
hr {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 20px 0;
}

/* Dataframe adapté au thème sombre */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e293b;
}

/* Expander et blocs */
div[data-testid="stExpander"] {
    background: #1c2541;
    border: 1px solid #1e293b !important;
    border-radius: 10px;
}

/* Cacher menu Streamlit */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Thème Plotly Harmonisé (Fonds Bleu Nuit & Textes Lumineux) ───────────────
# ─── Thème Plotly Harmonisé (Fonds Bleu Nuit & Textes Lumineux) ───────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", # Transparent pour adopter le bleu nuit du fond de l'app
    plot_bgcolor="rgba(0,0,0,0)",  # Zone de tracé transparente
    font=dict(
        family="Inter", 
        color="#e2e8f0",           # Texte clair (Gris-blanc) pour être très lisible sur bleu nuit
        size=11
    ),
    margin=dict(l=50, r=15, t=40, b=50), 
    legend=dict(
        bgcolor="rgba(28, 37, 65, 0.9)", # Fond de légende assorti aux cartes
        bordercolor="#1e293b", 
        borderwidth=1,
        font=dict(color="#e2e8f0")
    ),
    xaxis=dict(
        gridcolor="#1e293b",       # Lignes de grille sombres et discrètes
        linecolor="#334155",       # Ligne de l'axe visible mais douce
        tickfont=dict(color="#94a3b8"), # Chiffres de l'axe clairs
        title=dict(font=dict(color="#e2e8f0")) # <-- Correction ici : structure correcte pour le titre de l'axe
    ),
    yaxis=dict(
        gridcolor="#1e293b", 
        linecolor="#334155", 
        tickfont=dict(color="#94a3b8"), 
        title=dict(font=dict(color="#e2e8f0")) # <-- Correction ici aussi
    ),
)

CMAP = {"Churned": "#ff5a5f", "Retained": "#06d6a0"}

# ──────────────────────────────────────────────────────────────────────────────
# Exécution du notebook rh.ipynb
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_notebook(nb_path, csv_path):
    with open(nb_path) as f:
        nb = json.load(f)

    ns = {"__builtins__": __builtins__}
    df_avant_encodage = None

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).strip()
        if not src:
            continue

        src_exec = src.replace("dataset_rh_churn_renomme.csv", csv_path)

        if "LabelEncoder" in src and "df" in ns:
            df_avant_encodage = ns["df"].copy()

        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(compile(src_exec, "<notebook>", "exec"), ns)
        except Exception:
            pass
        finally:
            sys.stdout = old

    df_enc = ns.get("df", pd.DataFrame())
    if df_avant_encodage is None:
        df_avant_encodage = df_enc.copy()

    return df_enc, df_avant_encodage


df_encode, df = run_notebook("rh.ipynb", "dataset_rh_churn_renomme.csv")
df["churn_label"] = df["churn"].map({"Yes": "Churned", "No": "Retained"})

df["departement"]  = df["departement"].fillna("Non renseigné")
df["sexe"]         = df["sexe"].fillna("Non renseigné")
df["niveau_etude"] = df["niveau_etude"].fillna("Non renseigné")

NUM_COLS = ["age", "anciennete_annees", "salaire_mensuel_k",
            "heures_travail_hebdo", "score_performance", "absences_annuelles"]
LABELS = {
    "age":                  "Âge",
    "anciennete_annees":    "Ancienneté (ans)",
    "salaire_mensuel_k":    "Salaire (k€)",
    "heures_travail_hebdo": "Heures / semaine",
    "score_performance":    "Score performance",
    "absences_annuelles":   "Absences / an",
}

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation + filtres
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Analyse RH")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠  Aperçu général",
    "👤  Profil des employés",
    "🏢  Analyse par département",
    "📋  Données & statistiques",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")

dept_list  = sorted(df["departement"].unique())
dept_sel   = st.sidebar.multiselect("Département", dept_list, default=dept_list)

sexe_list  = sorted(df["sexe"].unique())
sexe_sel   = st.sidebar.multiselect("Sexe", sexe_list, default=sexe_list)

etude_list = sorted(df["niveau_etude"].unique())
etude_sel  = st.sidebar.multiselect("Niveau d'étude", etude_list, default=etude_list)

churn_sel  = st.sidebar.multiselect("Statut churn", ["Yes", "No"], default=["Yes", "No"])

age_min = int(df["age"].min())
age_max = int(df["age"].max())
age_range = st.sidebar.slider("Tranche d'âge", age_min, age_max, (age_min, age_max))

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset : {len(df)} lignes après nettoyage")

# ── Appliquer les filtres ─────────────────────────────────────────────────────
df_f = df[
    df["departement"].isin(dept_sel) &
    df["sexe"].isin(sexe_sel) &
    df["niveau_etude"].isin(etude_sel) &
    df["churn"].isin(churn_sel) &
    df["age"].between(age_range[0], age_range[1])
]

total   = len(df_f)
n_ch    = int((df_f["churn"] == "Yes").sum())
n_ret   = total - n_ch
taux    = round(n_ch / total * 100, 1) if total > 0 else 0
age_moy = round(df_f["age"].mean(), 1) if total > 0 else 0
sal_moy = round(df_f["salaire_mensuel_k"].mean(), 1) if total > 0 else 0


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Aperçu général
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠  Aperçu général":
    st.title("Aperçu général")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total employés",  total)
    c2.metric("Churned",         n_ch)
    c3.metric("Taux de churn",   f"{taux}%")
    c4.metric("Âge moyen",       f"{age_moy} ans")
    c5.metric("Salaire moyen",   f"{sal_moy} k€")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Répartition Churned / Retained")
        fig = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[n_ch, n_ret],
            hole=0.55,
            marker=dict(colors=["#ef4444", "#10b981"],
                        line=dict(color="white", width=2))
        ))
        fig.update_layout(**PL, height=340, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Churn par département")
        g = df_f.groupby(["departement", "churn_label"]).size().reset_index(name="n")
        fig = px.bar(g, x="departement", y="n", color="churn_label",
                     barmode="group", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=340, legend_title_text="",
                          xaxis_title="", yaxis_title="Effectif")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Taux de churn par département")
        g = (df_f.groupby("departement")["churn"]
               .apply(lambda x: (x == "Yes").mean() * 100)
               .reset_index(name="taux")
               .sort_values("taux", ascending=True))
        fig = px.bar(g, x="taux", y="departement", orientation="h",
                     color="taux",
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                     text=g["taux"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PL, height=300, coloraxis_showscale=False,
                          xaxis_title="Taux (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Taux de churn par niveau d'étude")
        g = (df_f.groupby("niveau_etude")["churn"]
               .apply(lambda x: (x == "Yes").mean() * 100)
               .reset_index(name="taux")
               .sort_values("taux", ascending=False))
        fig = px.bar(g, x="niveau_etude", y="taux",
                     color="taux",
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                     text=g["taux"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(**PL, height=300, coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Taux (%)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Profil des employés
# ──────────────────────────────────────────────────────────────────────────────
elif page == "👤  Profil des employés":
    st.title("Profil des employés")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution de l'âge")
        fig = px.histogram(df_f, x="age", color="churn_label",
                           nbins=25, barmode="overlay", opacity=0.7,
                           color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, legend_title_text="",
                          xaxis_title="Âge", yaxis_title="Fréquence")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribution du salaire mensuel")
        fig = px.histogram(df_f, x="salaire_mensuel_k", color="churn_label",
                           nbins=25, barmode="overlay", opacity=0.7,
                           color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, legend_title_text="",
                          xaxis_title="Salaire (k€)", yaxis_title="Fréquence")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Score de performance par churn")
        fig = px.box(df_f, x="churn_label", y="score_performance",
                     color="churn_label", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, showlegend=False,
                          xaxis_title="", yaxis_title="Score performance")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Absences annuelles par churn")
        fig = px.box(df_f, x="churn_label", y="absences_annuelles",
                     color="churn_label", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, showlegend=False,
                          xaxis_title="", yaxis_title="Absences / an")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Churn par sexe")
        g = df_f.groupby(["sexe", "churn_label"]).size().reset_index(name="n")
        fig = px.bar(g, x="sexe", y="n", color="churn_label",
                     barmode="group", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, legend_title_text="",
                          xaxis_title="", yaxis_title="Effectif")
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.subheader("Distribution de l'ancienneté")
        fig = px.histogram(df_f, x="anciennete_annees", color="churn_label",
                           nbins=20, barmode="overlay", opacity=0.7,
                           color_discrete_map=CMAP)
        fig.update_layout(**PL, height=310, legend_title_text="",
                          xaxis_title="Ancienneté (ans)", yaxis_title="Fréquence")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Analyse par département
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🏢  Analyse par département":
    st.title("Analyse par département")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Salaire moyen par département")
        g = (df_f.groupby(["departement", "churn_label"])["salaire_mensuel_k"]
               .mean().reset_index())
        fig = px.bar(g, x="departement", y="salaire_mensuel_k",
                     color="churn_label", barmode="group",
                     color_discrete_map=CMAP)
        fig.update_layout(**PL, height=320, legend_title_text="",
                          xaxis_title="", yaxis_title="Salaire moy. (k€)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Heures de travail par département")
        fig = px.box(df_f, x="departement", y="heures_travail_hebdo",
                     color="churn_label", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=320, legend_title_text="",
                          xaxis_title="", yaxis_title="Heures / semaine")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Score performance vs absences annuelles")
        s = df_f.sample(min(500, len(df_f)), random_state=42)
        fig = px.scatter(s, x="score_performance", y="absences_annuelles",
                         color="churn_label", opacity=0.65,
                         color_discrete_map=CMAP)
        fig.update_layout(**PL, height=320, legend_title_text="",
                          xaxis_title="Score performance",
                          yaxis_title="Absences / an")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Ancienneté moyenne par niveau d'étude")
        g = (df_f.groupby(["niveau_etude", "churn_label"])["anciennete_annees"]
               .mean().reset_index())
        fig = px.bar(g, x="niveau_etude", y="anciennete_annees",
                     color="churn_label", barmode="group",
                     color_discrete_map=CMAP)
        fig.update_layout(**PL, height=320, legend_title_text="",
                          xaxis_title="", yaxis_title="Ancienneté moy. (ans)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Matrice de corrélation")
    corr = df_f[NUM_COLS].corr().round(2)
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1)
    fig.update_layout(**PL, height=400)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Données & statistiques
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📋  Données & statistiques":
    st.title("Données & statistiques")
    st.markdown("---")

    st.subheader("Statistiques descriptives")
    st.dataframe(
        df_f[NUM_COLS].describe().round(2).rename(columns=LABELS),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader(f"Aperçu des données nettoyées — {total} lignes filtrées")
    st.dataframe(
        df_f.drop(columns=["churn_label"]).reset_index(drop=True).head(30),
        use_container_width=True
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Valeurs manquantes")
        miss = df_f.drop(columns=["churn_label"]).isna().sum().reset_index()
        miss.columns = ["Colonne", "Valeurs manquantes"]
        if miss["Valeurs manquantes"].sum() == 0:
            st.success("✅ Aucune valeur manquante après nettoyage.")
        else:
            st.dataframe(miss[miss["Valeurs manquantes"] > 0],
                         use_container_width=True)

    with col_b:
        st.subheader("Répartition du churn")
        vc = df_f["churn"].value_counts().reset_index()
        vc.columns = ["Churn", "Nombre"]
        vc["Pourcentage"] = (vc["Nombre"] / vc["Nombre"].sum() * 100).round(1)
        st.dataframe(vc, use_container_width=True)