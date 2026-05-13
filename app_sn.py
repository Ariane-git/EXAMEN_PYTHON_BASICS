"""
RH Churn Dashboard
Le preprocessing vient entièrement du notebook rh.ipynb :
  - L'app lit le fichier .ipynb
  - Exécute chaque cellule dans un namespace partagé (= kernel Jupyter)
  - Capture le df AVANT encodage  → pour les visualisations
  - Capture le df APRÈS encodage  → affiché comme sortie finale du notebook
Aucune logique de preprocessing n'est réécrite ici.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, io, sys, traceback, warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RH Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a3e 50%, #0d1a2e 100%);
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,.04) !important;
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #dde !important; }

/* ── KPI cards ── */
.kpi {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    transition: transform .2s, background .2s;
}
.kpi:hover { transform: translateY(-3px); background: rgba(255,255,255,.09); }
.kpi-v { font-size: 1.85rem; font-weight: 700; margin: 0; line-height: 1.1; }
.kpi-l { font-size: .68rem; text-transform: uppercase; letter-spacing: 1.1px;
         opacity: .5; margin-top: 5px; }
.kpi-d { font-size: .72rem; opacity: .4; margin-top: 3px; }

/* ── Section title ── */
.sec {
    font-size: .95rem; font-weight: 600; color: #c8c8e8;
    margin-bottom: 10px; padding-bottom: 7px;
    border-bottom: 1px solid rgba(255,255,255,.08);
}

/* ── Colors ── */
.r { color: #ff6b9d; } .g { color: #43e97b; }
.b { color: #667eea; } .o { color: #fdcb6e; } .p { color: #a29bfe; }

/* ── Misc ── */
#MainMenu, footer, header { visibility: hidden; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.04); border-radius: 10px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; color: rgba(255,255,255,.5) !important;
    font-weight: 500; font-size: .87rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(102,126,234,.35) !important; color: white !important;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Thème Plotly commun
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#aaa8c8", size=11),
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)",
               zerolinecolor="rgba(255,255,255,.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)",
               zerolinecolor="rgba(255,255,255,.06)"),
)
CR = "#ff6b9d"   # Churned
CG = "#43e97b"   # Retained
CMAP = {"Churned": CR, "Retained": CG}

NUM_COLS = ["age", "anciennete_annees", "salaire_mensuel_k",
            "heures_travail_hebdo", "score_performance", "absences_annuelles"]
CAT_COLS = ["niveau_etude", "departement", "sexe"]
LABELS   = {
    "age":                  "Âge",
    "anciennete_annees":    "Ancienneté (ans)",
    "salaire_mensuel_k":    "Salaire (k€)",
    "heures_travail_hebdo": "Heures / semaine",
    "score_performance":    "Score performance",
    "absences_annuelles":   "Absences / an",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MOTEUR : exécute rh.ipynb cellule par cellule
#  - Capture df juste AVANT la cellule LabelEncoder → df_viz (lisible)
#  - Capture df après la dernière cellule           → df_enc (encodé)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_notebook(nb_path: str, csv_path: str):
    with open(nb_path) as f:
        nb = json.load(f)

    # Namespace partagé = l'équivalent du kernel Jupyter
    ns = {"__builtins__": __builtins__}

    df_viz = None   # snapshot avant encodage

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).strip()
        if not src:
            continue

        # Remplacer uniquement le nom du fichier CSV → chemin absolu
        src_exec = src.replace("dataset_rh_churn_renomme.csv", csv_path)

        # Juste avant la cellule d'encodage : sauvegarder df lisible
        if "LabelEncoder" in src and "df" in ns:
            df_viz = ns["df"].copy()

        # Exécuter la cellule
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(compile(src_exec, "<notebook>", "exec"), ns)
        except Exception:
            pass
        finally:
            sys.stdout = old

    df_enc = ns.get("df", pd.DataFrame()).copy()

    # Fallback : si LabelEncoder n'a pas été trouvé
    if df_viz is None:
        df_viz = df_enc.copy()

    return df_enc, df_viz


# ─── Chargement ───────────────────────────────────────────────────────────────
NB_PATH  = "rh.ipynb"
CSV_PATH = "dataset_rh_churn_renomme.csv"

with st.spinner("⚙️  Exécution du notebook rh.ipynb…"):
    df_enc, df_viz = run_notebook(NB_PATH, CSV_PATH)

# Colonne churn lisible pour les visuels
df_viz["churn_label"] = df_viz["churn"].map({"Yes": "Churned", "No": "Retained"})


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filtres")
    st.markdown("---")

    churn_filt = st.multiselect(
        "Statut churn", ["Churned", "Retained"], default=["Churned", "Retained"])

    dept_opts = sorted(df_viz["departement"].dropna().unique())
    dept_filt = st.multiselect("Département", dept_opts, default=dept_opts)

    sexe_opts = sorted(df_viz["sexe"].dropna().unique())
    sexe_filt = st.multiselect("Sexe", sexe_opts, default=sexe_opts)

    etude_opts = sorted(df_viz["niveau_etude"].dropna().unique())
    etude_filt = st.multiselect("Niveau d'étude", etude_opts, default=etude_opts)

    age_min, age_max = int(df_viz["age"].min()), int(df_viz["age"].max())
    age_range = st.slider("Âge", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.markdown("**📓 Notebook chargé**")
    st.markdown(f"- Lignes nettoyées : **{len(df_viz):,}**")
    st.markdown(f"- Colonnes : **{df_viz.shape[1]}**")
    st.markdown(f"- Encodé : **{df_enc.shape}**")

# Appliquer les filtres
mask = (
    df_viz["churn_label"].isin(churn_filt)
    & df_viz["departement"].isin(dept_filt)
    & df_viz["sexe"].isin(sexe_filt)
    & df_viz["niveau_etude"].isin(etude_filt)
    & df_viz["age"].between(age_range[0], age_range[1])
)
dff = df_viz[mask]


# ─── Titre ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:14px 0 4px 0;'>
  <h1 style='font-size:2rem; font-weight:700; margin:0;
     background:linear-gradient(135deg,#667eea,#fdcb6e,#ff6b9d);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📊 RH Churn Analytics Dashboard
  </h1>
  <p style='opacity:.4; font-size:.8rem; margin-top:4px;'>
    Données issues de <b>rh.ipynb</b> · preprocessing exécuté à la volée
  </p>
</div>
""", unsafe_allow_html=True)


# ─── KPIs ─────────────────────────────────────────────────────────────────────
n_tot  = len(dff)
n_ch   = int((dff["churn_label"] == "Churned").sum())
n_ret  = n_tot - n_ch
rate   = n_ch / n_tot * 100 if n_tot else 0

def kpi(col, val, label, detail, color_cls):
    col.markdown(
        f'<div class="kpi">'
        f'<p class="kpi-v {color_cls}">{val}</p>'
        f'<p class="kpi-l">{label}</p>'
        f'<p class="kpi-d">{detail}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

k = st.columns(5)
kpi(k[0], f"{n_tot:,}",                       "Employés",          "après filtres",        "b")
kpi(k[1], f"{rate:.1f}%",                     "Taux de churn",     f"{n_ch:,} départs",    "r")
kpi(k[2], f"{n_ret:,}",                       "Retenus",           "en poste",             "g")
kpi(k[3], f"{dff['age'].mean():.1f} ans",     "Âge moyen",         "post-IQR notebook",    "o")
kpi(k[4], f"{dff['salaire_mensuel_k'].mean():.1f} k", "Salaire moyen", "mensuel (k€)",     "p")

st.markdown("<br>", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "  📈 Vue Générale  ",
    "  👤 Profil Employé  ",
    "  💼 Analyse RH  ",
    "  🔍 Exploration  ",
    "  📋 Données  ",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Vue Générale
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    c1, c2, c3 = st.columns([1.1, 1.6, 1.3])

    # Donut churn
    with c1:
        st.markdown('<p class="sec">Répartition Churn</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[n_ch, n_ret],
            hole=0.62,
            marker=dict(colors=[CR, CG],
                        line=dict(color="rgba(0,0,0,.3)", width=2)),
            textinfo="percent",
            textfont_size=13,
        ))
        fig.add_annotation(
            text=f"<b>{rate:.1f}%</b><br><span style='font-size:11px'>Churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="white"),
        )
        fig.update_layout(**PL, height=280, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Churn par département
    with c2:
        st.markdown('<p class="sec">Churn par Département</p>', unsafe_allow_html=True)
        g = dff.groupby(["departement", "churn_label"]).size().reset_index(name="n")
        fig = px.bar(g, x="departement", y="n", color="churn_label",
                     barmode="group", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=280, legend_title_text="",
                          xaxis_title="", yaxis_title="Effectif")
        st.plotly_chart(fig, use_container_width=True)

    # Churn par sexe
    with c3:
        st.markdown('<p class="sec">Churn par Sexe</p>', unsafe_allow_html=True)
        g = dff.groupby(["sexe", "churn_label"]).size().reset_index(name="n")
        fig = px.bar(g, x="sexe", y="n", color="churn_label",
                     barmode="stack", color_discrete_map=CMAP)
        fig.update_layout(**PL, height=280, legend_title_text="",
                          xaxis_title="", yaxis_title="Effectif")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c4, c5 = st.columns(2)

    # Taux churn / département
    with c4:
        st.markdown('<p class="sec">Taux de churn par Département</p>',
                    unsafe_allow_html=True)
        g = (dff.groupby("departement")["churn_label"]
               .apply(lambda x: (x == "Churned").mean() * 100)
               .reset_index(name="churn_pct")
               .sort_values("churn_pct", ascending=True))
        fig = px.bar(g, x="churn_pct", y="departement", orientation="h",
                     color="churn_pct",
                     color_continuous_scale=["#43e97b", "#fdcb6e", "#ff6b9d"],
                     text=g["churn_pct"].apply(lambda x: f"{x:.1f}%"))
        fig.update_layout(**PL, height=260,
                          coloraxis_showscale=False, xaxis_title="Taux (%)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Taux churn / niveau étude
    with c5:
        st.markdown('<p class="sec">Taux de churn par Niveau d\'Étude</p>',
                    unsafe_allow_html=True)
        g = (dff.groupby("niveau_etude")["churn_label"]
               .apply(lambda x: (x == "Churned").mean() * 100)
               .reset_index(name="churn_pct")
               .sort_values("churn_pct", ascending=False))
        fig = px.bar(g, x="niveau_etude", y="churn_pct",
                     color="churn_pct",
                     color_continuous_scale=["#43e97b", "#fdcb6e", "#ff6b9d"],
                     text=g["churn_pct"].apply(lambda x: f"{x:.1f}%"))
        fig.update_layout(**PL, height=260,
                          coloraxis_showscale=False, xaxis_title="",
                          yaxis_title="Taux (%)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Profil Employé
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    # Histogrammes par paire
    pairs = [(NUM_COLS[i], NUM_COLS[i + 1]) for i in range(0, len(NUM_COLS) - 1, 2)]
    for col_a, col_b in pairs:
        ca, cb = st.columns(2)
        for ax, cn in [(ca, col_a), (cb, col_b)]:
            ax.markdown(f'<p class="sec">{LABELS.get(cn, cn)}</p>',
                        unsafe_allow_html=True)
            fig = px.histogram(dff, x=cn, color="churn_label",
                               nbins=28, barmode="overlay", opacity=0.72,
                               color_discrete_map=CMAP,
                               labels={cn: LABELS.get(cn, cn)})
            fig.update_layout(**PL, height=240, legend_title_text="")
            ax.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec">Boxplots — variables numériques vs Churn</p>',
                unsafe_allow_html=True)
    bcols = st.columns(3)
    for i, cn in enumerate(NUM_COLS):
        with bcols[i % 3]:
            fig = px.box(dff, x="churn_label", y=cn, color="churn_label",
                         color_discrete_map=CMAP,
                         labels={cn: LABELS.get(cn, cn)})
            fig.update_layout(**PL, height=250,
                              showlegend=False, xaxis_title="",
                              yaxis_title=LABELS.get(cn, cn))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – Analyse RH
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    c1, c2 = st.columns(2)

    # Salaire moyen / département / churn
    with c1:
        st.markdown('<p class="sec">Salaire moyen par Département & Churn</p>',
                    unsafe_allow_html=True)
        g = (dff.groupby(["departement", "churn_label"])["salaire_mensuel_k"]
               .mean().reset_index())
        fig = px.bar(g, x="departement", y="salaire_mensuel_k",
                     color="churn_label", barmode="group",
                     color_discrete_map=CMAP,
                     labels={"salaire_mensuel_k": "Salaire moy. (k€)"})
        fig.update_layout(**PL, height=290,
                          legend_title_text="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Score performance vs absences
    with c2:
        st.markdown('<p class="sec">Score Performance vs Absences annuelles</p>',
                    unsafe_allow_html=True)
        s = dff.sample(min(800, len(dff)), random_state=42)
        fig = px.scatter(s, x="score_performance", y="absences_annuelles",
                         color="churn_label", opacity=0.55,
                         color_discrete_map=CMAP,
                         labels={"score_performance": "Score performance",
                                 "absences_annuelles": "Absences / an"})
        fig.update_layout(**PL, height=290, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    # Ancienneté / niveau étude
    with c3:
        st.markdown('<p class="sec">Ancienneté moyenne par Niveau d\'Étude & Churn</p>',
                    unsafe_allow_html=True)
        g = (dff.groupby(["niveau_etude", "churn_label"])["anciennete_annees"]
               .mean().reset_index())
        fig = px.bar(g, x="niveau_etude", y="anciennete_annees",
                     color="churn_label", barmode="group",
                     color_discrete_map=CMAP,
                     labels={"anciennete_annees": "Ancienneté moy. (ans)"})
        fig.update_layout(**PL, height=270,
                          legend_title_text="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Heures / semaine par département
    with c4:
        st.markdown('<p class="sec">Heures travail / semaine par Département</p>',
                    unsafe_allow_html=True)
        fig = px.violin(dff, x="departement", y="heures_travail_hebdo",
                        color="churn_label", box=True,
                        color_discrete_map=CMAP,
                        labels={"heures_travail_hebdo": "Heures / semaine"})
        fig.update_layout(**PL, height=270,
                          legend_title_text="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Matrice de corrélation
    st.markdown('<p class="sec">Matrice de corrélation (variables numériques)</p>',
                unsafe_allow_html=True)
    corr = dff[NUM_COLS].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale=["#ff6b9d", "#1a1a3e", "#43e97b"],
        zmin=-1, zmax=1,
        labels=dict(color="Corrélation"),
    )
    fig.update_layout(**PL, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – Exploration libre
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<p class="sec">🔍 Exploration personnalisée</p>',
                unsafe_allow_html=True)

    ea, eb, ec = st.columns(3)
    x_col = ea.selectbox("Axe X", NUM_COLS,
                          format_func=lambda c: LABELS.get(c, c), index=0)
    y_col = eb.selectbox("Axe Y", NUM_COLS,
                          format_func=lambda c: LABELS.get(c, c), index=2)
    ctype = ec.selectbox("Type", ["Scatter", "Boxplot", "Histogramme", "Violin"])

    s2 = dff.sample(min(2000, len(dff)), random_state=7)

    if ctype == "Scatter":
        fig = px.scatter(s2, x=x_col, y=y_col, color="churn_label",
                         opacity=0.5, color_discrete_map=CMAP,
                         labels={x_col: LABELS.get(x_col, x_col),
                                 y_col: LABELS.get(y_col, y_col)})
    elif ctype == "Boxplot":
        fig = px.box(dff, x="churn_label", y=x_col, color="churn_label",
                     color_discrete_map=CMAP,
                     labels={x_col: LABELS.get(x_col, x_col)})
    elif ctype == "Histogramme":
        fig = px.histogram(dff, x=x_col, color="churn_label",
                           nbins=30, barmode="overlay", opacity=0.75,
                           color_discrete_map=CMAP,
                           labels={x_col: LABELS.get(x_col, x_col)})
    else:
        fig = px.violin(dff, x="churn_label", y=x_col, color="churn_label",
                        box=True, color_discrete_map=CMAP,
                        labels={x_col: LABELS.get(x_col, x_col)})

    fig.update_layout(**PL, height=380, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec">Statistiques descriptives</p>',
                unsafe_allow_html=True)
    st.dataframe(
        dff[NUM_COLS].describe().round(2).rename(columns=LABELS),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – Données
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="sec">Données nettoyées (avant encodage)</p>',
                    unsafe_allow_html=True)
        st.dataframe(dff.drop(columns=["churn_label"]).reset_index(drop=True),
                     use_container_width=True, height=380)

    with c2:
        st.markdown('<p class="sec">Données encodées — sortie finale du notebook</p>',
                    unsafe_allow_html=True)
        st.dataframe(df_enc.reset_index(drop=True),
                     use_container_width=True, height=380)

    st.markdown("---")
    st.markdown('<p class="sec">Valeurs manquantes après nettoyage</p>',
                unsafe_allow_html=True)
    miss = dff.drop(columns=["churn_label"]).isna().sum().reset_index()
    miss.columns = ["Colonne", "Manquantes"]
    missing_cols = miss[miss["Manquantes"] > 0]
    if missing_cols.empty:
        st.success("✅ Aucune valeur manquante — nettoyage du notebook appliqué avec succès.")
    else:
        st.dataframe(missing_cols, use_container_width=True)
