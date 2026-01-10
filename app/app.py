import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# Wczytanie danych
df_bielik_polish = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\polish\bielik_polish_extracted_checked.csv', sep=';')
df_mistral_polish = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\polish\mistral_polish_extracted_checked.csv', sep=';')
df_gpt5_polish = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\polish\gpt5_polish_extracted_checked.csv', sep=';')
df_bielik_polish_rag = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\polish\rag\bielik_polish_rag_checked.csv', sep=';')
df_mistral_polish_rag = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\polish\rag\mistral_polish_rag_checked.csv', sep=';')
df_mistral_english_rag = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\rag\mistral_english_rag_checked.csv', sep=';')
df_mistral_english = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\mistral_english_extracted_checked.csv', sep=';')
df_deepseek_english_rag = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\rag\deepseek_english_rag_checked.csv', sep=';')
df_deepseek_english = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\deepseek_english_extracted_checked.csv', sep=';')
df_llama_english_rag = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\rag\llama_english_rag_checked.csv', sep=';')
df_llama_english = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\llama3_english_extracted_checked.csv', sep=';')
df_gpt5_english = pd.read_csv(r'C:\Users\ulasz\OneDrive\Pulpit\studia\GenderBias\results\english\gpt5_english_extracted_checked.csv', sep=';')


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
def plot_gender_by_job(df, title, ax, job_type_filter="All"):
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
            df[df["job_type"] == job_type_map[job_type_filter]]
            ["job_title_english"]
            .unique()
            .tolist()
        )

        jobs_list = [job for job in jobs_list if job in allowed_jobs]

    pivot = df.pivot_table(
        index="job_title_english",
        columns="gender_from_desc",
        aggfunc="size",
        fill_value=0
    )

    # 🔹 zachowanie kolejności + tylko wybrane zawody
    pivot = pivot.reindex(jobs_list).dropna(how="all")

    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "male/female": "#AB4444",
    }

    pivot.plot(
        kind="barh",
        stacked=True,
        legend=False,
        ax=ax,
        color=[colors.get(col, "gray") for col in pivot.columns]
    )

    ax.set_title(title, fontsize=17)
    ax.set_xlabel("Number of Responses", fontsize=14)
    ax.set_ylabel("Profession", fontsize=14)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)


# Funkcja do rysowania wykresu sankey'a
def plot_gender_sankey(
    df,
    job_title=None,
    title=""
):
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
    target_idx = {
        g: i + len(source_nodes)
        for i, g in enumerate(target_nodes)
    }

    sources = flow["gender_from_desc"].map(source_idx).tolist()
    targets = flow["gender_from_prompt"].map(target_idx).tolist()
    values = flow["count"].tolist()

    color_map = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#BBD67C",
        "male/female": "#AB4444",
        "invalid": "#292424"
    }

    node_colors = [
        color_map.get(label, "#7A337F")
        for label in labels
    ]

    link_colors = [
        color_map.get(g, "#CCCCCC")
        for g in flow["gender_from_desc"]
    ]

    fig = go.Figure(
        go.Sankey(
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
            )
        )
    )
    fig.update_traces(textfont_size=12, textfont_color="black")


    fig.update_layout(
        title_text=title,
        font_size=11,
        font_color="black"
    )

    return fig


def plot_neutral_delta_count_per_job(delta, title="Effect of RAG on Neutral Predictions per Job"):
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
        title=title,
        xaxis_title="Δ Neutral Count (RAG − No-RAG)",
        yaxis_title="Job Title",
        height=500,
        coloraxis_showscale=False,
        font=dict(size=14, color="black"),
        xaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black")),
        yaxis=dict(title=dict(font=dict(color="black")), tickfont=dict(color="black"))
    )

    return fig


# Streamlit App

st.set_page_config(
    layout="wide",
    page_title="Gender Analysis Dashboard"
)

st.sidebar.title("Filters")


language = st.sidebar.selectbox(
    "Language",
    ["PL", "EN", "FR"]
)
job_type_filter = st.sidebar.selectbox(
    "Job group",
    ["All", "Female-dominated", "Male-dominated", "Neutral"]
)


models = sorted(
    {model for (model, lang, _) in dataframes.keys() if lang == language}
)

st.title("Comparison of RAG vs No-RAG by Model")

for model in models:
    df_no_rag = dataframes.get((model, language, False))
    df_rag    = dataframes.get((model, language, True))

    if df_no_rag is None or df_rag is None:
        st.warning(f"No data for model {model} ({language})")
        continue
    
    df_no_rag = filter_by_job_type(df_no_rag, job_type_filter)
    df_rag    = filter_by_job_type(df_rag, job_type_filter)
    
    if df_no_rag.empty or df_rag.empty:
        st.info(f"No data for selected job group ({job_type_filter})")
        continue

    st.subheader(f"Model: {model}")
    st.markdown(
        "The bar charts below show the distribution of predicted genders "
        "across different professions for the selected model, comparing results "
        "without RAG (left) and with RAG (right). Colors correspond to genders."
    )


    fig, axes = plt.subplots(1, 2, figsize=(21, 10), sharey=True)

    plot_gender_by_job(df_no_rag, "Without RAG", axes[0])
    plot_gender_by_job(df_rag, "With RAG", axes[1])

    handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Gender",
        loc="center right",
        fontsize=15,
        title_fontsize=17
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    st.pyplot(fig)

    show_details = st.toggle(
    f"Show more detailed analysis for {model}",
    key=f"sankey_{model}")
    

    if show_details:
        job_options = ["All"] + sorted(
            set(df_no_rag["job_title_english"]) | set(df_rag["job_title_english"])
        )

        selected_job = st.selectbox(
            "Select job title",
            job_options,
            key=f"job_{model}"
        )
        st.subheader("Flow from Description-Derived Gender to Model-Predicted Gender")
        st.markdown(
            "These Sankey diagrams visualize how the predicted gender from the model "
            "flows from the description-derived gender. Left: without RAG, Right: with RAG."
        )
        col1, col2 = st.columns(2)

        with col1:
            # st.markdown("##### Without RAG")
            sankey_no_rag = plot_gender_sankey(
                df_no_rag,
                selected_job,
                title="Without RAG"
            )
            st.plotly_chart(
                sankey_no_rag,
                use_container_width=True
            )

        with col2:
            # st.markdown("##### With RAG")
            sankey_rag = plot_gender_sankey(
                df_rag,
                selected_job,
                title="With RAG"
            )
            st.plotly_chart(
    
                sankey_rag,
                use_container_width=True
            )

        delta_neutral_jobs = neutralization_delta_per_job(df_no_rag, df_rag)
        st.subheader("Change in Neutral Predictions per Job")
        st.markdown(
            "This bar chart shows how many additional 'neutral' predictions "
            "RAG produced for each profession compared to no-RAG. Bars to the right "
            "of the zero line indicate an increase in neutral predictions."
        )

        fig_jobs = plot_neutral_delta_count_per_job(
            delta_neutral_jobs,
            title=f"Effect of RAG on Neutral Predictions per Job – {model} ({language})"
        )

        st.plotly_chart(fig_jobs, use_container_width=True)



                                

            




