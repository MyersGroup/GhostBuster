import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib as mpl
from make_poplabels_dict import make_poplabels_dict

sns.set_palette("colorblind")
mpl.rc("font", **{"family": "normal", "size": 22})
plt.rc("axes.spines", **{"bottom": True, "left": True, "right": False, "top": False})
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2

wafr_pops = ["yoruba", "esn", "gwd", "msl", "mandenka", "I10871"]
rainforest_hg_pops = ["mbutipygmy", "biakapygmy"]
eafr_pops = ["lwk", "bantukenya", "Somali", "Luo", "Masai", "Dinka", "I5950"]
safr = [
    "san",
    "Ju_hoan_North",
    "Khomani_San",
    "bantusafrica",
    "ela001",
    "new001",
    "baa001",
]
ancients = ["I10871", "I5950", "ela001", "new001", "baa001"]


def collect_props():
    pop_labels = [
        "esn",
        "gwd",
        "msl",
        "lwk",
        "mbutipygmy",
        "biakapygmy",
        "san",
        "mandenka",
        "bantusafrica",
        "bantukenya",
        "yoruba",
        "I10871",
        "I5950",
        "ela001",
        "new001",
        "baa001",
        "Somali",
        "Luo",
        "Masai",
        "Ju_hoan_North",
        "Dinka",
        "Khomani_San",
    ]
    out = {p: [] for p in pop_labels}
    for pop in pop_labels:
        if pop in [
            "esn",
            "gwd",
            "msl",
            "lwk",
            "mbutipygmy",
            "biakapygmy",
            "san",
            "mandenka",
            "bantusafrica",
            "bantukenya",
            "yoruba",
        ]:
            prefix = "data/hgdp_1gp_v3/deepadmix_nohmm_upto300k/"
        elif pop in ancients:
            prefix = "data/hgdp_1gp_ancients/deepadmix_nohmm_upto300k/"
        else:
            prefix = "data/SGDP_only_moderns/deepadmix_nohmm_upto300k/"
        for file in glob.glob(
            prefix + str(pop) + "_overall_membership_*_sample_id_*.csv"
        ):
            if pop in [
                "Somali",
                "Luo",
                "Masai",
                "Ju_hoan_North",
                "Dinka",
                "Khomani_San",
            ]:
                file_runall = file.replace(
                    pop + "_overall_membership", "SGDP_specific_afr_overall_membership"
                )
                df = pd.read_csv(file_runall, sep="\t")
            else:
                df = pd.read_csv(file, sep="\t")
            df_bta = (
                pd.read_csv(
                    file.replace("deepadmix_nohmm_upto300k/", "back_to_africa/"),
                    sep="\t",
                )
                .rename(columns={"prob_0": "prob_afr", "prob_1": "prob_eur"})
                .drop(columns=["genpos"])
            )
            mdf = pd.merge(df, df_bta, on=["chr", "pos"])
            mdf = mdf.loc[mdf["prob_afr"] > 0.95]
            r = (mdf["genpos"].shift(2) - mdf["genpos"].shift(-2)) / (
                mdf["pos"].shift(2) - mdf["pos"].shift(-2)
            )
            mdf = mdf.loc[r < r.quantile(0.5)]
            out[pop].append(
                100 * min((mdf["prob_1"] > 0.5).mean(), (mdf["prob_0"] > 0.5).mean())
            )
    return out


props = collect_props()
groups = {
    **{p: "West Africans" for p in wafr_pops},
    **{p: "Rainforest HG" for p in rainforest_hg_pops},
    **{p: "East Africans" for p in eafr_pops},
    **{p: "South Africans" for p in safr},
}
region_colors = {
    "West Africans": "#1f77b4",
    "Rainforest HG": "#2ca02c",
    "East Africans": "#ff7f0e",
    "South Africans": "#8c564b",
}
ancient_gray = "0.6"

poplabels_dict = make_poplabels_dict()

summaries = {p: np.median(v) for p, v in props.items() if len(v) > 0}
pop_order = sorted(summaries.keys(), key=lambda p: summaries[p])
labels = [poplabels_dict.get(p, p) for p in pop_order]

records = []
for pop in pop_order:
    for val in props[pop]:
        records.append(
            {"population": pop, "label": poplabels_dict.get(pop, pop), "prop": val}
        )
df = pd.DataFrame(records)
print(df)

plt.clf()
fig, ax = plt.subplots(figsize=(7, 8))
palette = {}
for p in pop_order:
    palette[poplabels_dict.get(p, p)] = region_colors[groups[p]]

# sns.boxplot(
#     x="prop", y="label",
#     data=df,
#     order=labels,
#     linewidth=1,
#     notch=True, showcaps=False,
#     flierprops={"marker":"x"},
#     width=0.65,
#     ax=ax,
#     palette=palette
# )
sns.violinplot(
    x="prop",
    y="label",
    data=df,
    ax=ax,
    orient="h",
    inner=None,
    linewidth=1.0,
    palette=palette,
    alpha=0.8,
)
sns.swarmplot(
    data=df,
    x="prop",
    y="label",
    ax=ax,
    orient="h",
    color="k",
    size=3,
    edgecolor="black",
    alpha=0.25,
)
for i, artist in enumerate(ax.artists):
    p = pop_order[i]
    c = region_colors[groups[p]]
    artist.set_facecolor(c)
    artist.set_edgecolor("k")
    artist.set_alpha(1.0)

for line in ax.lines:
    line.set_color("k")

for tick, p in zip(ax.get_yticklabels(), pop_order):
    if p in ancients:
        tick.set_fontsize(12)  # concrete integer font size
        tick.set_color("red")
    else:
        tick.set_fontsize(15)  # keep default size for non-ancients

ax.set_xlabel("OOA-like proportion (%)", fontsize=18)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("ooa_props.svg", dpi=300)

legend_handles = [
    Patch(facecolor=region_colors[g], edgecolor="k", label=g, alpha=0.8)
    for g in ["West Africans", "Rainforest HG", "East Africans", "South Africans"]
]
fig_leg, ax_leg = plt.subplots(figsize=(5.2, 2.0))
ax_leg.axis("off")
ax_leg.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    ncol=4,
    fontsize=14,
    handlelength=1.0,
    handletextpad=0.5,
)
# ensure layout and avoid clipping
fig_leg.tight_layout()
fig_leg.savefig("ooa_props_legend.svg", dpi=300, bbox_inches="tight")

means = {p: np.mean(v) for p, v in props.items() if len(v) > 0}

for p in sorted(means, key=means.get):
    print(f"{p}\t{means[p]:.3f}")
