import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
print(os.getcwd())
# Wczytanie danych
df_bielik_polish = pd.read_csv("results/polish/bielik_polish_extracted_checked.csv", sep=';')
df_mistral_polish = pd.read_csv("results/polish/mistral_polish_extracted_checked.csv", sep=';')
df_gpt5_polish = pd.read_csv("results/polish/gpt5_polish_extracted_checked.csv", sep=';')
df_bielik_polish_rag = pd.read_csv("results/polish/rag/bielik_polish_rag_checked.csv", sep=';')
df_mistral_polish_rag = pd.read_csv("results/polish/rag/mistral_polish_rag_checked.csv", sep=';')
df_mistral_english_rag = pd.read_csv("results/english/rag/mistral_english_rag_checked.csv", sep=';')
df_mistral_english = pd.read_csv("results/english/mistral_english_extracted_checked.csv", sep=';')
df_deepseek_english_rag = pd.read_csv("results/english/rag/deepseek_english_rag_checked.csv", sep=';')
df_deepseek_english = pd.read_csv("results/english/deepseek_english_extracted_checked.csv", sep=';')
df_llama_english_rag = pd.read_csv("results/english/rag/llama_english_rag_checked.csv", sep=';')
df_llama_english = pd.read_csv("results/english/llama3_english_extracted_checked.csv", sep=';')
df_gpt5_english = pd.read_csv("results/english/gpt5_english_extracted_checked.csv", sep=';')




# funkcja do filtrowania ramek danych według grupy zawodów
def filter_by_job_type(df, job_type_filter):
    if job_type_filter == "All":
        return df

    mapping = {
        "Female-dominated": "female dominated",
        "Male-dominated": "male dominated",
        "Neutral": "neutral"

    }

    return df[df["job_type"] == mapping[job_type_filter]]

# funkcje do obliczania wskaźników neutralności
def neutral_count_per_job(df):
    """
    Dla każdego zawodu liczymy liczbę przypadków, gdzie gender_from_prompt == 'neutral'
    """
    return (
        df[df["gender_from_desc"] == "neutral"]
          .groupby("job_title_english")
          .size()
          .reset_index(name="neutral_count")
    )


def neutralization_delta_per_job(df_no_rag, df_rag):
    nr_no = neutral_count_per_job(df_no_rag)
    nr_rag = neutral_count_per_job(df_rag)

    delta = nr_no.merge(
        nr_rag,
        on="job_title_english",
        how='outer',
        suffixes=("_no_rag", "_rag")
    ).fillna(0)

    # Delta = wzrost neutralnych dzięki RAG
    delta["delta_neutral"] = delta["neutral_count_rag"] - delta["neutral_count_no_rag"]
    return delta




# mapowanie ramek danych
dataframes = {
    ("Bielik", "PL", False): df_bielik_polish,
    ("Mistral", "PL", False): df_mistral_polish,
    ("Bielik", "PL", True): df_bielik_polish_rag,
    ("Mistral", "PL", True): df_mistral_polish_rag,
    ("Mistral", "EN", False): df_mistral_english,
    ("Mistral", "EN", True): df_mistral_english_rag,
    ("Deepseek", "EN", False): df_deepseek_english,
    ("Deepseek", "EN", True): df_deepseek_english_rag,
    ("Llama", "EN", False): df_llama_english,
    ("Llama", "EN", True): df_llama_english_rag
}



def plot_gender_by_job_plotly_combined(df_no_rag, df_rag, job_type_filter="All", model_name="", language=""):
    jobs_list = [
        "secretary", "dressmaker", "nurse", "psychologist", "librarian",
        "HR specialist", "dietician", "school teacher", "cosmetologist",
        "speech therapist", "software engineer", "firefighter", "carpenter",
        "taxi driver", "aircraft pilot", "mechanical engineer",
        "chief executives", "miner", "mathematician", "fisher",
        "accountant", "judge", "pharmacist",
        "financial analyst", "dining room staff"
    ]

    # 🔹 filtrowanie po job_type
    if job_type_filter != "All":
        job_type_map = {
            "Female-dominated": "female dominated",
            "Male-dominated": "male dominated",
            "Neutral": "neutral"
        }
        allowed_jobs = (
            df_no_rag[df_no_rag["job_type"] == job_type_map[job_type_filter]]
            ["job_title_english"]
            .unique()
            .tolist()
        )
        jobs_list = [job for job in jobs_list if job in allowed_jobs]

    def prepare_pivot(df):
        pivot = df.pivot_table(
            index="job_title_english",
            columns="gender_from_desc",
            aggfunc="size",
            fill_value=0
        )
        pivot = pivot.reindex(jobs_list).dropna(how="all")
        return pivot

    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "male/female": "#9779AD",
        "non-binary": "#FFE27B",
        "invalid": "#EEE7E7"
    }

    pivot_no_rag = prepare_pivot(df_no_rag)
    pivot_rag = prepare_pivot(df_rag)

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.15,  # odstęp między wykresami
        subplot_titles=("Without RAG", "With RAG")
    )
# Lista wszystkich genderów występujących w obu wykresach
    all_genders = list(set(pivot_no_rag.columns) | set(pivot_rag.columns))

    # Trzymamy info, które gendery już dodaliśmy do legendy
    legend_shown = {gender: False for gender in all_genders}

    # 🔹 Wykres Without RAG
    for gender in pivot_no_rag.columns:
        fig.add_trace(
            go.Bar(
                y=pivot_no_rag.index,
                x=pivot_no_rag[gender],
                name=gender,
                orientation='h',
                marker_color=colors.get(gender, "black"),
                showlegend=not legend_shown[gender]  # pokaż legendę tylko raz
            ),
            row=1, col=1
        )
        legend_shown[gender] = True  # oznaczamy, że legendę dodano

    # 🔹 Wykres With RAG
    for gender in pivot_rag.columns:
        fig.add_trace(
            go.Bar(
                y=pivot_rag.index,
                x=pivot_rag[gender],
                name=gender,
                orientation='h',
                marker_color=colors.get(gender, "black"),
                showlegend=not legend_shown[gender]  # tylko te, które jeszcze nie były w legendzie
            ),
            row=1, col=2
        )
        legend_shown[gender] = True


    fig.update_layout(
    barmode='stack',
    title=dict(
        text=f"Gender Distribution by Job Title - {model_name} ({language})",
        font=dict(size=20, color="black"),
        x=0.5,  # środkowanie tytułu głównego
        xanchor="center"
    ),
    font=dict(size=14, color="black"),
    height=600,
    width=1400,
    xaxis=dict(
        title=dict(text="Number of Occurrences", font=dict(color="black", size=14)),
        tickfont=dict(color="black", size=12)
    ),
    xaxis2=dict(
        title=dict(text="Number of Occurrences", font=dict(color="black", size=14)),
        tickfont=dict(color="black", size=12)
    ),
    yaxis=dict(
        title=dict(text="Job Title", font=dict(color="black", size=14)),
        tickfont=dict(color="black", size=12)
    ),
    yaxis2=dict(
        tickfont=dict(color="black", size=12)
    ),
    legend=dict(
        title=dict(text="Gender", font=dict(size=12, color="black")),
        font=dict(size=12, color="black"),
        x=1.02,  # przesunięcie w prawo
        y=0.5,   # wyśrodkowanie pionowe
        xanchor="left",
        yanchor="middle"
    ),
    margin=dict(l=150, r=150, t=100, b=50)
)


    return fig

# Funkcja do rysowania wykresu sankey'a
def plot_gender_sankey_subplots(df_no_rag, df_rag, job_title=None, model_name="", language=""):
    # Funkcja pomocnicza do stworzenia Sankey dla pojedynczego df
    def create_sankey(df, title=""):
        if job_title and job_title != "All":
            df = df[df["job_title_english"] == job_title]

        flow = (
            df.groupby(["gender_from_desc", "gender_from_prompt"])
            .size()
            .reset_index(name="count")
        )

        source_nodes = flow["gender_from_desc"].unique().tolist()
        target_nodes = flow["gender_from_prompt"].unique().tolist()
        labels = source_nodes + target_nodes

        source_idx = {g: i for i, g in enumerate(source_nodes)}
        target_idx = {g: i + len(source_nodes) for i, g in enumerate(target_nodes)}

        sources = flow["gender_from_desc"].map(source_idx).tolist()
        targets = flow["gender_from_prompt"].map(target_idx).tolist()
        values = flow["count"].tolist()

        color_map = {
            "female": "#F08080",
            "male": "#6495ED",
            "neutral": "#A9A9A9",
            "non-binary": "#FFE27B",
            "male/female": "#9779AD",
            "invalid": "#EEE7E7"
        }

        node_colors = [color_map.get(label, "#7A337F") for label in labels]
        link_colors = [color_map.get(g, "#CCCCCC") for g in flow["gender_from_desc"]]

        sankey = go.Sankey(
            node=dict(
                label=labels,
                color=node_colors,
                pad=15,
                thickness=20,
                line=dict(color="black")
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            ),
            textfont=dict(color="black", size=12)
        )

        return sankey, title

    # Tworzymy subploty z 1 wierszem i 2 kolumnami
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Without RAG", "With RAG"),
        specs=[[{"type": "domain"}, {"type": "domain"}]],  # dla Sankey
        horizontal_spacing=0.2
    )

    # Dodajemy Sankey do pierwszego wykresu
    sankey1, title1 = create_sankey(df_no_rag, "Without RAG")
    fig.add_trace(sankey1, row=1, col=1)

    # Dodajemy Sankey do drugiego wykresu
    sankey2, title2 = create_sankey(df_rag, "With RAG")
    fig.add_trace(sankey2, row=1, col=2)

    # Ustawienia layoutu
    fig.update_layout(
        title=dict(
            text=f"Gender Flow Sankey Diagram - {model_name} ({language})",
            font=dict(size=20, color="black"),
            x=0.5,  # środkowanie tytułu głównego
            xanchor="center"
        ),
        font=dict(size=12, color="black"),
        height=500,
        width=1000,
        showlegend=False  # Sankey ma własną legendę w node labels
    )

    return fig

def plot_overall_distribution_shared(df_left, df_right, model_name="", language=""):
    order = ["female", "male", "male/female", "non-binary", "neutral"]
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD",
    }

    def prep(df):
        c = df['gender_from_desc'].value_counts().reset_index()
        c.columns = ['gender', 'count']
        return c[c['count'] > 0]

    left = prep(df_left)
    right = prep(df_right)

    # 🔹 pełna pula kategorii z obu df
    all_genders = order  # zawsze bierzemy wszystkie możliwe kategorie, żeby legenda była kompletna
    legend_shown = {g: False for g in all_genders}  # śledzimy, które wartości dodano do legendy

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Without RAG", "With RAG"),
        shared_yaxes=True,
        horizontal_spacing=0.15
    )

    # 🔹 Wykres lewy
    for g in all_genders:
        if g in left.gender.values:
            fig.add_trace(
                go.Bar(
                    x=[g],
                    y=[left[left.gender == g]['count'].values[0]],
                    name=g,
                    marker_color=colors[g],
                    legendgroup=g,
                    showlegend=not legend_shown[g]
                ),
                row=1, col=1
            )
            legend_shown[g] = True

    # 🔹 Wykres prawy
    for g in all_genders:
        if g in right.gender.values:
            fig.add_trace(
                go.Bar(
                    x=[g],
                    y=[right[right.gender == g]['count'].values[0]],
                    name=g,
                    marker_color=colors[g],
                    legendgroup=g,
                    showlegend=False  # legendę pokazujemy tylko raz
                ),
                row=1, col=2
            )

    # 🔹 Ustawienia layoutu
    fig.update_layout(
        height=500,
        width=1200,
        barmode="group",
        xaxis=dict(title="Gender", tickfont=dict(color="black", size=12)),
        xaxis2=dict(title="Gender", tickfont=dict(color="black", size=12)),
        yaxis=dict(title="Count", tickfont=dict(color="black", size=12)),
        yaxis2=dict(tickfont=dict(color="black", size=12)),
        font=dict(size=14, color="black"),
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12, color="black"),
            title=dict(text="Gender", font=dict(size=12, color="black"))
        ),
        title=dict(
            text=f"Overall Gender Distribution - {model_name} ({language})",
            font=dict(size=20, color="black"),
            x=0.5,
            xanchor="center"
        )
    )

    return fig

def plot_overall_distribution(df, title, showlegend=True):
    counts = df['gender_from_desc'].value_counts().reset_index()
    counts.columns = ['gender', 'count']
    counts = counts[counts['count'] > 0]
    order = ["female", "male", "male/female", "non-binary", "neutral"]
    true_order = [g for g in order if g in counts['gender'].values]

    fig = px.bar(
        counts,
        x='gender',
        y='count',
        color='gender',
        color_discrete_map={
            "female": "#F08080",
            "male": "#6495ED",
            "neutral": "#A9A9A9",
            "non-binary": "#FFE27B",
            "male/female": "#9779AD",
        },
        category_orders={"gender": true_order}
    )

    fig.update_layout(
        title=title,
        xaxis_title="Gender",
        yaxis_title="Count",
        height=400,
        coloraxis_showscale=False,
        font=dict(size=14, color="black"),
        xaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black")),
        yaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black"))
    )

    return fig

def plot_neutral_delta_count_per_job(delta, title="Effect of RAG on Neutral Descriptions per Job"):
    jobs_list = ["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                 "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                 "taxi driver", "aircraft pilot", "mechanical engineer", "chief executives", "miner", "mathematician",
                 "fisher", "accountant", "judge", "pharmacist", "financial analyst", "dining room staff"]

    delta["job_title_english"] = pd.Categorical(delta["job_title_english"], categories=jobs_list, ordered=True)
    delta = delta.sort_values("job_title_english")

    fig = px.bar(
        delta,
        x="delta_neutral",
        y="job_title_english",
        orientation="h",
        color_discrete_sequence=["#A9A9A9"],  # jednolity kolor słupków
    )

    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color="black"),
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Δ Neutral Count (RAG − No-RAG)",
        yaxis_title="Job Title",
        height=500,
        coloraxis_showscale=False,
        font=dict(size=14, color="black"),
        xaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black")),
        yaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black"))
    )

    return fig
def render_model_analysis(model, language, job_type_filter):

    df_no_rag = dataframes.get((model, language, False))
    df_rag    = dataframes.get((model, language, True))

    if df_no_rag is None or df_rag is None:
        st.warning(f"No data for model {model} ({language})")
        return

    df_no_rag = filter_by_job_type(df_no_rag, job_type_filter)
    df_rag    = filter_by_job_type(df_rag, job_type_filter)

    if df_no_rag.empty or df_rag.empty:
        st.info(f"No data for selected job group ({job_type_filter})")
        return

    st.subheader(f"{model}")

    # -------- BAR CHARTS --------
    # st.markdown("#### Gender Distribution by Job Title")
    fig = plot_gender_by_job_plotly_combined(df_no_rag, df_rag, job_type_filter, model_name=model, language=language)
    st.plotly_chart(fig, use_container_width=True)



    # -------- OVERALL DISTRIBUTION --------
    # st.markdown("#### Overall Gender Distribution")
    st.plotly_chart(
    plot_overall_distribution_shared(df_no_rag, df_rag, model_name=model, language=language),
    use_container_width=True
)      


    # -------- DETAILS TOGGLE --------
    show_details = st.toggle(
        "Show more detailed analysis",
        key=f"sankey_{language}_{model}"
    )

    if show_details:

        job_options = ["All"] + sorted(
            set(df_no_rag["job_title_english"]) |
            set(df_rag["job_title_english"])
        )

        selected_job = st.selectbox(
            "Select job title",
            job_options,
            key=f"job_{language}_{model}"
        )
        # ------ SANKY DIAGRAMS --------
        fig = plot_gender_sankey_subplots(df_no_rag, df_rag, job_title=selected_job, model_name=model, language=language)
        st.plotly_chart(fig, use_container_width=True)

       
        # -------- DELTA NEUTRAL --------
        delta_neutral_jobs = neutralization_delta_per_job(df_no_rag, df_rag)

        st.plotly_chart(
            plot_neutral_delta_count_per_job(
                delta_neutral_jobs,
                title=f"Effect of RAG on Neutral Descriptions per Job – {model} ({language})"
            ),
            use_container_width=True
        )
# =========================
# STREAMLIT APP (DYNAMIC MODELS BY LANGUAGE)
# =========================

st.set_page_config(layout="wide", page_title="Gender Bias Analysis Dashboard")

st.title("Gender Bias Analysis Dashboard")

# ----------- FILTERS -----------
col_filters, col_main = st.columns([1, 5])

with col_filters:
    st.subheader("Filters")

    language = st.selectbox("Language", ["EN", "PL"], index=0)

    job_type_filter = st.selectbox(
        "Job group",
        ["All", "Female-dominated", "Male-dominated", "Neutral"]
    )

# ----------- MAIN VIEW -----------
with col_main:

    models = sorted({
        model for (model, lang, _) in dataframes.keys()
        if lang == language
    })

    if not models:
        st.warning("No models available for this language.")
        st.stop()

    model_tabs = st.tabs(models)

    for tab, model in zip(model_tabs, models):
        with tab:
            render_model_analysis(model, language, job_type_filter)

