"""Figuras del capítulo 6 (datos de las tablas del Anexo B)."""
import matplotlib as mpl
from matplotlib import font_manager as fm
import glob
for f in glob.glob("/usr/share/fonts/opentype/linux-libertine/*.otf"):
    fm.fontManager.addfont(f)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

mpl.rcParams.update({
    "font.family": "Linux Libertine O", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

# ── Figura 1: precisión frente a exhaustividad ────────────────────
# (nombre, prec, exh) sin post-proceso y con B2 — Tabla B.3
sin = {
    False: [("Baseline (s42)", .43, .57), ("Baseline (s43)", .36, .58),
            ("IoU", .37, .55), ("clDice", .44, .56), ("IoU+clDice", .46, .56)],
    True: [("Topograph (s42)", .49, .53), ("Topograph (s43)", .51, .49),
           ("IoU+Topo", .48, .50), ("clDice+Topo", .49, .51),
           ("IoU+clDice+Topo", .50, .51)],
}
b2 = {
    False: [("Baseline (s42)", .63, .53), ("Baseline (s43)", .61, .50),
            ("IoU", .58, .51), ("clDice", .57, .53), ("IoU+clDice", .59, .55)],
    True: [("Topograph (s42)", .57, .53), ("Topograph (s43)", .65, .47),
           ("IoU+Topo", .62, .46), ("clDice+Topo", .59, .50),
           ("IoU+clDice+Topo", .57, .49)],
}
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.9), sharex=True, sharey=True)
for ax, datos, tit in [(axs[0], sin, "Sin post-proceso"), (axs[1], b2, "Con B2")]:
    for topo, col, mk, lab in [(False, AZUL, "o", "Sin Topograph"),
                               (True, NARANJA, "s", "Con Topograph")]:
        e = [d[2] for d in datos[topo]]; p = [d[1] for d in datos[topo]]
        ax.scatter(e, p, s=34, c=col, marker=mk, edgecolors="white",
                   linewidths=1.2, label=lab, zorder=3)
    ax.set_title(tit, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("Exhaustividad")
    ax.xaxis.set_major_formatter(coma); ax.yaxis.set_major_formatter(coma)
    ax.set_xlim(.44, .60); ax.set_ylim(.33, .68)
axs[0].set_ylabel("Precisión")
axs[0].legend(loc="upper right", frameon=False, fontsize=9, handletextpad=.3)
fig.tight_layout()
fig.savefig("fig_precision_exhaustividad.pdf")

# ── Figura 2: tipos de error, baseline frente a Topograph ─────────
# (partidos+unidos, sin pred., sin fil.) — Tabla B.2
mods = ["Baseline\ns42", "Baseline\ns43", "Topograph\ns42", "Topograph\ns43"]
err = {
    "Sin post-proceso": [(104, 140, 319), (125, 100, 482), (91, 183, 169), (81, 224, 110)],
    "Con B2": [(55, 250, 68), (50, 275, 88), (53, 217, 108), (40, 300, 49)],
}
fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
for ax, (tit, vals) in zip(axs, err.items()):
    x = range(len(mods)); base = [0] * len(mods)
    for k, (lab, col) in enumerate([("Sin filamento", AZUL),
                                    ("Sin predicción", NARANJA),
                                    ("Partidos + unidos", AQUA)]):
        h = [v[[2, 1, 0][k]] for v in vals]
        ax.bar(x, h, bottom=base, width=.62, color=col, label=lab,
               edgecolor="white", linewidth=1.5)
        base = [b + hh for b, hh in zip(base, h)]
    ax.set_xticks(list(x)); ax.set_xticklabels(mods, fontsize=8)
    ax.set_title(tit, fontsize=10, color="#0b0b0b")
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Número de errores")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig("fig_tipos_error.pdf")
