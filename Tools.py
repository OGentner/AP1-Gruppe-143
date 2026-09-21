# todo 
# latext ausgabe vom fehler und nur eine gute Funktion machen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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

# daten, parameter = lade_versuch(
#     "reversionspendel.xlsx"
# )

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

def gauss_error(f, variables, errors):
    """
    Gaußsche Fehlerfortpflanzung:
    
    f         : Funktion/Ausdruck
    variables : Liste der Variablen [x, y, ...]
    errors    : Liste der Unsicherheiten [dx, dy, ...]
    """
    sigma = 0

    for var, error in zip(variables, errors):
        sigma += (sp.diff(f, var) * error)**2

    return sp.sqrt(sigma)

def gauss_error_values(f, variables, values, errors):
    substitutions = dict(zip(variables, values))

    # Funktionswert
    f_value = float(f.subs(substitutions))

    # Gaußsche Fehlerfortpflanzung
    error = 0
    for var, err in zip(variables, errors):
        derivative = sp.diff(f, var)
        error += (derivative.subs(substitutions) * err)**2
        print((derivative.subs(substitutions) * err)**2)

    # error_tex = 0
    # for var, err in zip(variables, errors):
    #     derivative = sp.diff(f, var)
    #     error_tex += (derivative*err)**2
    # print(sp.latex((error_tex)))

    error_value = float(sp.sqrt(error))

    return f_value, error_value

def zTest(Bestwert:float, Literaturwert:float, Standardunsicherheit:float, LaTexAusgabe=False):
    """
    Berechnet den z-Wert für die gegebenen Werte.
    Ausgabe hat keine Einheiten :(
    """
    z = abs((Bestwert - Literaturwert) / Standardunsicherheit)

    if LaTexAusgabe:
        if z > 2:
            print(f"Da $z = \left|\\frac{{\hat x-y}}{{\Delta x}}\\right| = \left|\\frac{{ {Bestwert:.2f} - {Literaturwert:.2f}}}{{{Standardunsicherheit:.2f}}} \\right| = {z:.2f} > 2$, weicht unser Ergebnis Signifikant vom Literaturwert ab.")
        else: print(f"Da $z = \left|\\frac{{\hat x-y}}{{\Delta x}}\\right| = \left|\\frac{{ {Bestwert:.2f} - {Literaturwert:.2f}}}{{{Standardunsicherheit:.2f}}}\\right| = {z:.2f} <= 2$, ist unser Ergebnis mit dem Literaturwert (${Literaturwert:.2f}$) kompatibel.")
    return z


# s, T0 = sp.symbols('s T0')
# g = 4*s*sp.pi**2/T0**2
# dg = gauss_error(
#     g,
#     [s, T0],
#     [sp.Symbol('ds'), sp.Symbol('dT0')]
# )

# print(dg)

# Fehlerfortpflanzung Volumen
print("Volumen 1:", end=" ")
h, k, d1, d2 = sp.symbols('h k d1 d2')
V = h*k*(d1+d2)/2
V1 = gauss_error_values(
    V,
    [h, k, d1, d2],
    [5.07, 23.04, 34.4, 34.95],
    [0.005, 0.005, 0.05, 0.05]
)
print(V1)

print("Volumen 2:", end=" ")
r,d = sp.symbols('r d')
V = sp.pi*(r**2)*0.5*d
V2 = gauss_error_values(
    V,
    [r, d],
    [22/2, 32],
    [0.05, 0.05])
print(V2)


print("Masse 1:", end=" ")
x, a, z = sp.symbols('x a z')
m = (x - a)/z
m1  = gauss_error_values(
    m,
    [x, a, z],
    [0.246, 0, (400/82)],
    [0.0003, 0.0005, 0.02])
print(m1)

print("Masse 2:", end=" ")
x, a, z = sp.symbols('x a z')
m = (x - a)/z
m2 = gauss_error_values(
    m,
    [x, a, z],
    [0.214, 0, (400/82)],
    [0.0003, 0.0005, 0.02])
print(m2)


print("Dichte 1:", end=" ")
m, v = sp.symbols('m v')
d = m/v
d1 = gauss_error_values(
    d,
    [m, v],
    [m1[0], V1[0]*1e-9],
    [m1[1], V1[1]*1e-9])
print(d1)

# zTest(d1[0], 12450, d1[1], LaTexAusgabe=True)

print("Dichte 2:", end=" ")
m, v = sp.symbols('m v')
d = m/v
d2 = gauss_error_values(
    d,
    [m, v],
    [m2[0], V2[0]*1e-9],
    [m2[1], V2[1]*1e-9])
print(d2)






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

