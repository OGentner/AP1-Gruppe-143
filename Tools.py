import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

# s = daten["Länge s [m]"].to_numpy()

# messzeiten = daten[
#     ["T1 [s]", "T2 [s]", "T3 [s]", "T4 [s]", "T5 [s]"]
# ].to_numpy()

def lade_versuch(datei):

    messdaten = pd.read_excel(
        datei,
        sheet_name="Messdaten"
    )

    parameter = pd.read_excel(
        datei,
        sheet_name="Parameter"
    )

    return messdaten, parameter

daten, parameter = lade_versuch(
    "reversionspendel.xlsx"
)

# latex(1+2**(x+y)) #SymPy x, y müssen Type "symbols" sein
# (Latex(r"$A = P(1 + r)^t$"))
# (Latex(f"$A = {P:,}(1 + {r})^{{{t}}}$"))
# (Latex(f"$A = {A:,.2f}$"))


def makeLaTexTable(array, labels, caption, tag):
    """test = np.vstack([np.round(Stossparameter_Zwei, 2), np.round(Theta_Zwei, 2), np.round(Stdabw_Zwei, 2)])
print(makeLaTexTable(test, [r"$b$ (cm)", r"$\theta$", r"$s_\theta$"], "Streuwinkel und ...", "Winkel_2"))"""
    latex = f"""\\begin{{table}}[H]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|{"l" * array.shape[1]}}}\n"""

    for row in range(array.shape[0]):
        latex += f"\\textbf{{{labels[row]}}}"
        for column in range(array.shape[1]):
            latex += "& "
            latex += str(array[row, column])
            latex += "\t"

        if row < array.shape[0]:
            latex += "\\\\ \n"
        else: 
            latex += "\n"
    latex += f"""
\\end{{tabular}}%
}}
\\caption{{{caption}}}
\\label{{tab:{tag}}}
\\end{{table}}"""

    return latex



# fig, ax = plt.subplots(dpi=600)
# ax.errorbar(x, y, Fehler, fmt='.', linewidth=2, capsize=6)
# ax.set_title("Titel")
# ax.grid()
# plt.show()

# # Residuen errechnen:)
# m = 0.0315  # Steigung
# t = 0       # Verschiebung
# vals = x * m + t

# # Diagramm mit Ausgleichsgerade
# fig, ax = plt.subplots(dpi=600)
# ax.errorbar(x, y, Fehler, fmt='.', linewidth=2, capsize=6)
# ax.plot(x, vals)
# ax.set_title("Ausgleichsgerade")
# ax.grid()
# plt.show()

# # Residuendiagramm
# diff = y - vals
# fig, ax = plt.subplots(dpi=600)
# ax.plot(x, diff, '.')
# ax.grid()
# ax.set_title("Res-Diagramm")
# plt.show()

