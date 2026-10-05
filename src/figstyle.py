import matplotlib as mpl, matplotlib.pyplot as plt
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", violet="#4a3aa7", grey="#898781", ink="#0b0b0b", sec="#52514e")
METH = {"persist": ("Persistence", C["grey"], "o", ":"), "ar": ("AR", C["yellow"], "s", "--"), "tf": ("TF (routing)", C["blue"], "^", "--"),
        "lgbm": ("LightGBM", C["orange"], "v", "-."), "gru": ("Vanilla GRU", C["magenta"], "D", "-"), "full": ("PG-FaultTwin", C["violet"], "o", "-")}
REG = {"MidAtlantic": ("Mid-Atlantic", C["blue"]), "OhioValley": ("Ohio Valley", C["orange"]), "TexasHill": ("Texas Hill Country", C["aqua"]), "UpperMidwest": ("Upper Midwest", C["magenta"])}
def setup():
    mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#898781",
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "#fcfcfb", "figure.facecolor": "white",
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
        "text.color": C["ink"], "axes.labelcolor": C["sec"], "xtick.color": C["sec"], "ytick.color": C["sec"], "font.family": "DejaVu Sans"})
