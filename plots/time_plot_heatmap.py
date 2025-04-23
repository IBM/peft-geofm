# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager

fname = "/Library/Fonts/IBM-Plex-Sans/IBMPlexSans-Regular.otf"  # download IBM Plex Sans
font_manager.fontManager.addfont(fname)
prop = font_manager.FontProperties(fname=fname)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 28,
        "figure.titlesize": 28,
    }
)
# %%
RED_CMAP = LinearSegmentedColormap.from_list(
    "red_ibm",
    colors=[  # red 20 to red 60
        "#ffd7d9",
        # "#ffb3b8",
        "#ff8389",
        # "#fa4d56",
        # "#da1e28",
        "#a2191f",
    ],
)

GREEN_CMAP = LinearSegmentedColormap.from_list(
    "green_ibm",
    colors=[  # teal 20 to green 60
        "#9ef0f0",
        # "#3ddbd9",
        # "#08bdba",
        # "#009d9a",
        "#007d79",
    ],
)
#  %%
df_speed = pd.read_csv("../data/epoch_and_time.csv", header=[0, 1, 2], index_col=[0, 1])
df_miou = pd.read_csv("../data/peft_table.csv", header=[0, 1, 2], index_col=[0, 1])


def create_plot(include_prithvi_2_100, df_speed, df_miou):
    models = (
        [
            "DeCUR",
            "Clay v1",
            "Prithvi 1.0 100M",
            "Prithvi 2.0 100M",
            "Prithvi 2.0 300M",
        ]
        if include_prithvi_2_100
        else [
            "DeCUR",
            "Clay v1",
            "Prithvi 1.0 100M",
            "Prithvi 2.0 300M",
        ]
    )

    methods = ["Linear Prob.", "VPT", "LoRA", "ViT Adapter", "Full FT"]
    df_speed = df_speed.loc[models]
    df_miou = df_miou.loc[models]

    miou = df_miou[("test mIoU", "mean")]
    speed = df_speed[("total_time", "mean")]
    miou_mean = miou.mean(axis=1)
    speed_mean = speed.mean(axis=1)
    # epoch = df_speed[("metrics.epoch", "mean")]
    # epoch_mean = epoch.mean(axis=1)

    miou_data = miou_mean.unstack(level=0).loc[methods]
    speed_data = speed_mean.unstack(level=0).loc[methods]

    fig, ax = plt.subplots(1, 2, figsize=(21 if include_prithvi_2_100 else 19, 6.5))
    sns.heatmap(miou_data, annot=True, fmt=".2f", cmap=GREEN_CMAP, ax=ax[0])
    sns.heatmap(speed_data, annot=True, fmt=".2f", cmap=RED_CMAP, ax=ax[1], vmax=35)

    for a in ax:
        a.set_facecolor("grey")
        a.set_xlabel("Model", labelpad=10)
        a.set_yticklabels(ax[1].get_yticklabels(), rotation=0)
        a.set_aspect(0.9)
    ax[0].set_ylabel("Method", labelpad=10)
    ax[1].set_ylabel("")
    ax[0].set_title("Average test mIoU (%)", pad=10)
    ax[1].set_title("Average training time (min)", pad=10)
    filename = (
        "time_miou_v2_all.pdf" if include_prithvi_2_100 else "time_miou_v2_paper.pdf"
    )
    plt.savefig(f"../assets/{filename}", pad_inches=0, bbox_inches="tight")


# %%
create_plot(True, df_speed, df_miou)
create_plot(False, df_speed, df_miou)

# %%
