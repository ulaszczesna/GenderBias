import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def clean_data(df):
    df['gender_from_prompt'] = df['gender_from_prompt'].str.strip()
    df['gender_from_desc'] = df['gender_from_desc'].str.strip()
    df.loc[df['gender_from_prompt'] == 'f', 'gender_from_prompt'] = 'female'
    df.loc[df['gender_from_prompt'] == 'm', 'gender_from_prompt'] = 'male'
    df.loc[df['gender_from_prompt'] == 'm/f', 'gender_from_prompt'] = 'male/female'
    df.loc[df['gender_from_desc'] == 'm/f', 'gender_from_desc'] = 'male/female'
    df.loc[df['gender_from_desc'] == 'nautral', 'gender_from_desc'] = 'neutral'
    df.loc[df['gender_from_prompt'] == 'n', 'gender_from_prompt'] = 'neutral'
    df.loc[df['gender_from_desc'] == 'other', 'gender_from_desc'] = 'neutral'
    df.loc[df['gender_from_prompt'] == 'f/non-bin', 'gender_from_prompt'] = 'female'
    df.loc[df['gender_from_prompt'] == 'm/non-bin', 'gender_from_prompt'] = 'male'
    df.loc[df['gender_from_prompt'] == 'femals', 'gender_from_prompt'] = 'female'
    df.loc[df['gender_from_desc'] == 'femals', 'gender_from_desc'] = 'female'
    if 'jobs_english' in df.columns:
        df['job_title_english'] = df.pop('jobs_english')
    df.loc[df['job_title_english'] == 'chief executives', 'job_title_english'] = 'chief executive'

    print("After cleaning:")
    print("Gender from description")
    print(df['gender_from_desc'].value_counts())
    print("Gender from prompt")
    print(df['gender_from_prompt'].value_counts())

    return df

def plot_gender_from_desc(df, title, output_filename, jobs_list=["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                         "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                         "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive", "miner", "mathematician",
                         "fisher", "accountant", "judge",
                         "pharmacist", "financial analyst", "dining room staff"]):
    pivot = df.pivot_table(
        index="job_title_english",
        columns="gender_from_desc",
        aggfunc="size",
        fill_value=0
    )
    pivot = pivot.reindex(jobs_list)
    gender_order = ["female", "male", "male/female", "non-binary", "neutral"]
    gender_order = [g for g in gender_order if g in pivot.columns]
    pivot = pivot[gender_order]
    
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B", 
        "male/female": "#9779AD"
    }

 
    pivot.plot(
        kind="barh",
        stacked=True,
        figsize=(10,8),
        color=[colors.get(col, "gray") for col in pivot.columns]
    )
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.title(title)


    plt.xticks(rotation=90)
    plt.ylabel("Job title")
    plt.xlabel("Number of Occurrences")
    plt.legend(
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5)   
    )

    plt.tight_layout()
    plt.savefig(f"praca/{output_filename}.png", transparent=True)
    plt.show()
    

def plot_gender_from_prompt(df, title,  output_filename, jobs_list=["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                         "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                         "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive", "miner", "mathematician",
                         "fisher", "accountant", "judge",
                         "pharmacist", "financial analyst", "dining room staff"]):

    pivot = df.pivot_table(
        index="job_title_english",
        columns="gender_from_prompt",
        aggfunc="size",
        fill_value=0
    )
    pivot = pivot.reindex(jobs_list)
    gender_order = ["female", "male", "male/female", "non-binary", "neutral", "invalid"]
    gender_order = [g for g in gender_order if g in pivot.columns]
    pivot = pivot[gender_order]
    
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD",
        "invalid": "#EEE7E7"
    }

  
    pivot.plot(
        kind="barh",
        stacked=True,
        figsize=(10,8),
        color=[colors.get(col, "gray") for col in pivot.columns]
    )
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.title(title)

    plt.xticks(rotation=90)
    plt.ylabel("Job title")
    plt.xlabel("Number of Occurrences")
    plt.legend(
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5)  
    )

    plt.tight_layout()
    plt.savefig(f"praca/{output_filename}.png")
    plt.show()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_three_datasets(dfs, title, output_filename, subtitles=None, jobs_list=None):
    
    if jobs_list is None:
        jobs_list = ["secretary", "dressmaker", "nurse", "psychologist", "librarian",
                     "HR specialist", "dietician", "school teacher", "cosmetologist",
                     "speech therapist", "software engineer", "firefighter", "carpenter",
                     "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive",
                     "miner", "mathematician", "fisher", "accountant", "judge", "pharmacist",
                     "financial analyst", "dining room staff"]
    
    if subtitles is None:
        subtitles = [f"Dataset {i+1}" for i in range(len(dfs))]
    
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD"
    }
    
    fig, axes = plt.subplots(1, len(dfs), figsize=(8*len(dfs), 8))  # Zwiększona szerokość z 7 na 9
    if len(dfs) == 1:
        axes = [axes]  # ensure axes is iterable
    
    all_genders_in_data = set()
    
    for i, df in enumerate(dfs):
        pivot = df.pivot_table(
            index="job_title_english",
            columns="gender_from_desc",
            aggfunc="size",
            fill_value=0
        )
        pivot = pivot.reindex(jobs_list)
        
        gender_order = ["female", "male", "male/female", "non-binary", "neutral"]
        gender_order = [g for g in gender_order if g in pivot.columns]
        pivot = pivot[gender_order]
        
        all_genders_in_data.update(gender_order)
        
        colors_list = [colors.get(col, "gray") for col in pivot.columns]
        
        pivot.plot(
            kind="barh",
            stacked=True,
            ax=axes[i],
            color=colors_list,
            legend=False
        )
        
        axes[i].set_xlabel("Number of Occurrences")
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].set_title(subtitles[i], fontsize=12)
        
        # Tylko pierwszy wykres ma etykiety osi Y
        if i == 0:
            axes[i].set_ylabel("Job title")
            axes[i].set_yticks(range(len(jobs_list)))
            axes[i].set_yticklabels(jobs_list)
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticks(range(len(jobs_list)))
            axes[i].set_yticklabels([])  # usuń napisy na osi Y
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Legenda dla kategorii które faktycznie występują
    gender_order_final = ["female", "male", "male/female", "non-binary", "neutral"]
    genders_to_show = [g for g in gender_order_final if g in all_genders_in_data]
    
    handles = [plt.Line2D([0], [0], color=colors[g], lw=8) for g in genders_to_show]
    labels = genders_to_show
    
    fig.legend(
        handles,
        labels,
        title="Gender",
        loc='center left',
        bbox_to_anchor=(0.89, 0.5)  # Przesunięcie legendy bliżej wykresów
    )
    
    # Tight layout ze zwiększonym wspace i dostosowanym rect
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # Zmniejszenie marginesu z prawej strony
    plt.subplots_adjust(wspace=0.03)  # Zwiększenie odstępu między wykresami
    
    # Save figure
    plt.savefig(f"praca/{output_filename}.png", transparent=True, bbox_inches='tight')
    plt.show()
def plot_overall_distribution_shared(dfs, model_names, output_filename, language=""):
    import matplotlib.pyplot as plt
    )
    PREFERRED_LEGEND_ORDER = ["female", "male", "male/female", "non-binary", "neutral"]
    
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

  
    prepared_data = [prep(df) for df in dfs]
    
   
    genders_present = []
    for data in prepared_data:
        genders_present.extend(list(data['gender']))
    genders_present = list(dict.fromkeys(genders_present))
    
    all_genders_legend = [g for g in PREFERRED_LEGEND_ORDER if g in genders_present]
    all_genders_legend += [g for g in genders_present if g not in PREFERRED_LEGEND_ORDER]


    max_count = 0
    for data in prepared_data:
        if len(data) > 0:
            max_count = max(max_count, data['count'].max())

    # ✅ Create figure with 1x3 layout (three plots side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes_flat = axes
    
  
    def plot_bars(ax, data, title):
      
        genders_in_data = list(data['gender'])
        genders_to_plot = [g for g in PREFERRED_LEGEND_ORDER if g in genders_in_data]
        genders_to_plot += [g for g in genders_in_data if g not in PREFERRED_LEGEND_ORDER]
        
        x_positions = range(len(genders_to_plot))
        counts = []
        colors_list = []
        
        for g in genders_to_plot:
            counts.append(data[data.gender == g]['count'].values[0])
            colors_list.append(colors.get(g, "#888"))
        
        ax.bar(x_positions, counts, color=colors_list)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(genders_to_plot, fontsize=12, color="black")
        ax.set_xlabel("Gender", fontsize=12, color="black")
        ax.set_ylabel("Count", fontsize=12, color="black")
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='y', labelsize=12, colors='black')
        ax.tick_params(axis='both', which='both', length=0)
        
        
        ax.set_ylim(0, max_count * 1.1)
        
       
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Plot all models
    for i, (data, model_name) in enumerate(zip(prepared_data, model_names)):
        plot_bars(axes_flat[i], data, model_name)
    

    fig.suptitle(f"Overall Gender Distribution - {language}", 
                 fontsize=20, color="black", y=0.98)
    
  
    handles = [plt.Line2D([0], [0], color=colors[g], lw=8) for g in all_genders_legend]
    labels = all_genders_legend
    
    fig.legend(
        handles,
        labels,
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
        title_fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(f"praca/{output_filename}.png")
    plt.show()


def plot_gender_comparison(df, model_name, language, output_filename, 
                          jobs_list=["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                                    "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                                    "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive", "miner", "mathematician",
                                    "fisher", "accountant", "judge",
                                    "pharmacist", "financial analyst", "dining room staff"]):
    

    pivot_desc = df.pivot_table(
        index="job_title_english",
        columns="gender_from_desc",
        aggfunc="size",
        fill_value=0
    )
    pivot_desc = pivot_desc.reindex(jobs_list)
    
 
    pivot_prompt = df.pivot_table(
        index="job_title_english",
        columns="gender_from_prompt",
        aggfunc="size",
        fill_value=0
    )
    pivot_prompt = pivot_prompt.reindex(jobs_list)
    
 
    gender_order_desc = ["female", "male", "male/female", "non-binary", "neutral"]
    gender_order_desc = [g for g in gender_order_desc if g in pivot_desc.columns]
    pivot_desc = pivot_desc[gender_order_desc]
    
    gender_order_prompt = ["female", "male", "male/female", "non-binary", "neutral", "invalid"]
    gender_order_prompt = [g for g in gender_order_prompt if g in pivot_prompt.columns]
    pivot_prompt = pivot_prompt[gender_order_prompt]
    

    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD",
        "invalid": "#EEE7E7"
    }
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
 
    pivot_desc.plot(
        kind="barh",
        stacked=True,
        ax=ax1,
        color=[colors.get(col, "gray") for col in pivot_desc.columns],
        legend=False
    )
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.set_title("Gender Representation in Job Descriptions - {model} {lang}".format(model=model_name, lang=language))
    ax1.set_ylabel("Job title")
    ax1.set_xlabel("Number of Occurrences")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.tick_params(axis='both', which='both', length=0)
    

    pivot_prompt.plot(
        kind="barh",
        stacked=True,
        ax=ax2,
        color=[colors.get(col, "gray") for col in pivot_prompt.columns],
        legend=False
    )
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title("Gender Representation in Direct Gender Prompts - {model} {lang}".format(model=model_name, lang=language))
    ax2.set_xlabel("Number of Occurrences")
    ax2.set_ylabel("Job title") 
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax2.tick_params(axis='both', which='both', length=0)
    
   
    all_genders = list(dict.fromkeys(list(pivot_desc.columns) + list(pivot_prompt.columns)))
    handles = [plt.Rectangle((0,0),1,1, color=colors.get(g, "gray")) for g in all_genders]
    
    fig.legend(
        handles,
        all_genders,
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(f"praca/{output_filename}.png", transparent=True, bbox_inches='tight')
    plt.show()

def plot_gender_comparison_with_rag(df_no_rag, df_rag, model_name, language, output_filename, 
                          jobs_list=["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                                    "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                                    "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive", "miner", "mathematician",
                                    "fisher", "accountant", "judge",
                                    "pharmacist", "financial analyst", "dining room staff"]):
  
    pivot_desc_no_rag = df_no_rag.pivot_table(
        index="job_title_english",
        columns="gender_from_desc",
        aggfunc="size",
        fill_value=0
    )
    pivot_desc_no_rag = pivot_desc_no_rag.reindex(jobs_list)
    
   
    pivot_prompt_no_rag = df_no_rag.pivot_table(
        index="job_title_english",
        columns="gender_from_prompt",
        aggfunc="size",
        fill_value=0
    )
    pivot_prompt_no_rag = pivot_prompt_no_rag.reindex(jobs_list)
    
 
    pivot_desc_rag = df_rag.pivot_table(
        index="job_title_english",
        columns="gender_from_desc",
        aggfunc="size",
        fill_value=0
    )
    pivot_desc_rag = pivot_desc_rag.reindex(jobs_list)
    

    pivot_prompt_rag = df_rag.pivot_table(
        index="job_title_english",
        columns="gender_from_prompt",
        aggfunc="size",
        fill_value=0
    )
    pivot_prompt_rag = pivot_prompt_rag.reindex(jobs_list)
    
  
    gender_order_desc = ["female", "male", "male/female", "non-binary", "neutral"]
    gender_order_prompt = ["female", "male", "male/female", "non-binary", "neutral", "invalid"]
    
    for pivot in [pivot_desc_no_rag, pivot_desc_rag]:
        existing = [g for g in gender_order_desc if g in pivot.columns]
        if existing:
            pivot_cols = pivot[existing]
            for col in existing:
                pivot[col] = pivot_cols[col]
    
    for pivot in [pivot_prompt_no_rag, pivot_prompt_rag]:
        existing = [g for g in gender_order_prompt if g in pivot.columns]
        if existing:
            pivot_cols = pivot[existing]
            for col in existing:
                pivot[col] = pivot_cols[col]
    
 
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD",
        "invalid": "#EEE7E7"
    }
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
  
    gender_order_desc_filtered = [g for g in gender_order_desc if g in pivot_desc_no_rag.columns]
    if gender_order_desc_filtered:
        pivot_desc_no_rag[gender_order_desc_filtered].plot(
            kind="barh",
            stacked=True,
            ax=ax1,
            color=[colors.get(col, "gray") for col in gender_order_desc_filtered],
            legend=False
        )
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.set_title("Gender Representation in Job Descriptions - {model} {lang}".format(model=model_name, lang=language))
    ax1.set_ylabel("Job title")
    ax1.set_xlabel("Number of Occurrences")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='both', which='both', length=0)
    
  
    gender_order_desc_filtered_rag = [g for g in gender_order_desc if g in pivot_desc_rag.columns]
    if gender_order_desc_filtered_rag:
        pivot_desc_rag[gender_order_desc_filtered_rag].plot(
            kind="barh",
            stacked=True,
            ax=ax2,
            color=[colors.get(col, "gray") for col in gender_order_desc_filtered_rag],
            legend=False
        )
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title("Gender Representation in Job Descriptions - {model} {lang} without RAG".format(model=model_name, lang=language))
    ax2.set_ylabel("Job title")
    ax2.set_xlabel("Number of Occurrences")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='both', which='both', length=0)
    
    gender_order_prompt_filtered = [g for g in gender_order_prompt if g in pivot_prompt_no_rag.columns]
    if gender_order_prompt_filtered:
        pivot_prompt_no_rag[gender_order_prompt_filtered].plot(
            kind="barh",
            stacked=True,
            ax=ax3,
            color=[colors.get(col, "gray") for col in gender_order_prompt_filtered],
            legend=False
        )
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.set_title("Gender Representation in Direct Gender Prompts - {model} {lang} with RAG".format(model=model_name, lang=language))
    ax3.set_ylabel("Job title")
    ax3.set_xlabel("Number of Occurrences")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(axis='both', which='both', length=0)
    

    gender_order_prompt_filtered_rag = [g for g in gender_order_prompt if g in pivot_prompt_rag.columns]
    if gender_order_prompt_filtered_rag:
        pivot_prompt_rag[gender_order_prompt_filtered_rag].plot(
            kind="barh",
            stacked=True,
            ax=ax4,
            color=[colors.get(col, "gray") for col in gender_order_prompt_filtered_rag],
            legend=False
        )
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax4.set_title("Gender Representation in Direct Gender Prompts - {model} {lang}".format(model=model_name, lang=language))
    ax4.set_ylabel("Job title")
    ax4.set_xlabel("Number of Occurrences")
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(axis='both', which='both', length=0)
    
 
    all_genders = list(dict.fromkeys(
        gender_order_desc_filtered + gender_order_desc_filtered_rag + 
        gender_order_prompt_filtered + gender_order_prompt_filtered_rag
    ))
    handles = [plt.Rectangle((0,0),1,1, color=colors.get(g, "gray")) for g in all_genders]
    
    fig.legend(
        handles,
        all_genders,
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(f"praca/{output_filename}.png", transparent=True, bbox_inches='tight')
    plt.show()


def plot_rag_comparison(df_no_rag, df_rag, model_name, language, output_filename, 
                         from_desc=True,
                         jobs_list=["secretary", "dressmaker", "nurse", "psychologist", "librarian", "HR specialist", "dietician", 
                                    "school teacher", "cosmetologist", "speech therapist", "software engineer", "firefighter", "carpenter",
                                    "taxi driver", "aircraft pilot", "mechanical engineer", "chief executive", "miner", "mathematician",
                                    "fisher", "accountant", "judge",
                                    "pharmacist", "financial analyst", "dining room staff"]):
    
    # Wybór bazowego tytułu na podstawie from_desc
    if from_desc:
        base_title = "results from descriptions"
        target_col = "gender_from_desc"
    else:
        base_title = "results from gender prompts"
        target_col = "gender_from_prompt"
    
    # Kolory i kolejność (identyczne z oryginałem)
    colors = {
        "female": "#F08080",
        "male": "#6495ED",
        "neutral": "#A9A9A9",
        "non-binary": "#FFE27B",
        "male/female": "#9779AD",
        "invalid": "#EEE7E7"
    }
    gender_order = ["female", "male", "male/female", "non-binary", "neutral", "invalid"]

    def prepare_pivot(df):
        pivot = df.pivot_table(index="job_title_english", columns=target_col, aggfunc="size", fill_value=0)
        pivot = pivot.reindex(jobs_list).fillna(0)
        cols = [g for g in gender_order if g in pivot.columns]
        return pivot[cols]

    pivot_left = prepare_pivot(df_no_rag)
    pivot_right = prepare_pivot(df_rag)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def apply_style(ax, pivot, title):
        pivot.plot(
            kind="barh", stacked=True, ax=ax,
            color=[colors.get(col, "gray") for col in pivot.columns],
            legend=False
        )
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_title(title)
        ax.set_ylabel("Job title")
        ax.set_xlabel("Number of Occurrences")
        # Usunięcie ramek dokładnie jak w oryginale
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)

    # Tytuły z dopiskiem na końcu
    title_left = f"No-RAG {base_title} – {model_name} {language}"
    title_right = f"RAG {base_title} – {model_name} {language}"

    apply_style(ax1, pivot_left, title_left)
    apply_style(ax2, pivot_right, title_right)

    # Legenda
    combined_cols = list(dict.fromkeys(list(pivot_left.columns) + list(pivot_right.columns)))
    sorted_legend = [g for g in gender_order if g in combined_cols]
    handles = [plt.Rectangle((0,0),1,1, color=colors.get(g, "gray")) for g in sorted_legend]
    
    fig.legend(handles, sorted_legend, title="Gender", loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(f"praca/{output_filename}.png", transparent=True, bbox_inches='tight')
    plt.show()


def plot_overall_distribution_shared_two(dfs, model_names, output_filename, language=""):

    
    # Preferred legend order (others appended)
    PREFERRED_LEGEND_ORDER = ["female", "male", "male/female", "non-binary", "neutral"]
    
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

    # Prepare all datasets
    prepared_data = [prep(df) for df in dfs]
    
    # Determine all genders present across all datasets (for legend)
    genders_present = []
    for data in prepared_data:
        genders_present.extend(list(data['gender']))
    genders_present = list(dict.fromkeys(genders_present))
    
    all_genders_legend = [g for g in PREFERRED_LEGEND_ORDER if g in genders_present]
    all_genders_legend += [g for g in genders_present if g not in PREFERRED_LEGEND_ORDER]

    # Find max count across all datasets for consistent y-axis
    max_count = 0
    for data in prepared_data:
        if len(data) > 0:
            max_count = max(max_count, data['count'].max())

    # ✅ Create figure with 1x2 layout (two plots side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes_flat = axes
    
    # Helper function to plot bars
    def plot_bars(ax, data, title):
        # Get genders present in THIS specific dataset
        genders_in_data = list(data['gender'])
        genders_to_plot = [g for g in PREFERRED_LEGEND_ORDER if g in genders_in_data]
        genders_to_plot += [g for g in genders_in_data if g not in PREFERRED_LEGEND_ORDER]
        
        x_positions = range(len(genders_to_plot))
        counts = []
        colors_list = []
        
        for g in genders_to_plot:
            counts.append(data[data.gender == g]['count'].values[0])
            colors_list.append(colors.get(g, "#888"))
        
        ax.bar(x_positions, counts, color=colors_list)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(genders_to_plot, fontsize=12, color="black")
        ax.set_xlabel("Gender", fontsize=12, color="black")
        ax.set_ylabel("Count", fontsize=12, color="black")
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='y', labelsize=12, colors='black')
        ax.tick_params(axis='both', which='both', length=0)
        
        # Set consistent y-axis limit
        ax.set_ylim(0, max_count * 1.1)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Plot all models
    for i, (data, model_name) in enumerate(zip(prepared_data, model_names)):
        plot_bars(axes_flat[i], data, model_name)
    
    # Main title
    fig.suptitle(f"Overall Gender Distribution - {language}", 
                 fontsize=20, color="black", y=0.98)
    
    # Create legend (with all genders that appear across any dataset)
    handles = [plt.Line2D([0], [0], color=colors[g], lw=8) for g in all_genders_legend]
    labels = all_genders_legend
    
    fig.legend(
        handles,
        labels,
        title="Gender",
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
        title_fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(f"praca/{output_filename}.png")
    plt.show()
