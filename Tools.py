import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# LaTex Formatierung in Plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

def lade_versuch(datei):
    """Lädt aus einer Excel-Datei die Daten aus den Tabellen "Messdaten" und "Parameter" und gibt diese als Type pd.DataFrame zurück.
    """

    messdaten = pd.read_excel(
        datei,
        sheet_name="Messdaten"
    )

    parameter = pd.read_excel(
        datei,
        sheet_name="Parameter"
    )

    return messdaten, parameter

def makeLaTexTable(array, labels, caption:str, tag:str):
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

def gauss_error_values(
    f,
    variables,
    values,
    errors,
    eqLabel="label",
    variable_name="f",
    unit="",
    returnLaTex=False
):
    """
    Gaußsche Fehlerfortpflanzung.

    f              : SymPy-Ausdruck
    variables      : Liste der Variablen [x, y, ...]
    values         : Liste der Messwerte [x, y, ...]
    errors         : Liste der Unsicherheiten [dx, dy, ...]
    eqLabel        : Label für die LaTeX-Gleichung
    variable_name  : Name der Ergebnisgröße, z.B. "m"
    unit           : Einheit, z.B. r"\\,\\mathrm{kg}"
    returnLaTex    : Gibt zusätzlich die LaTeX-Strings zurück
    """

    # Werte den Variablen zuordnen
    substitutions = dict(zip(variables, values))

    # --------------------------------------------------
    # 1. Funktionswert
    # --------------------------------------------------

    f_value = float(f.subs(substitutions))

    # --------------------------------------------------
    # 2. Gaußsche Fehlerfortpflanzung
    # --------------------------------------------------

    error_sum = 0

    # Terme der allgemeinen Gauß-Formel
    symbolic_terms = []

    # Terme mit berechneten symbolischen Ableitungen
    derivative_terms = []

    for var, err in zip(variables, errors):

        # Partielle Ableitung
        derivative = sp.diff(f, var)

        # Numerischer Wert der Ableitung
        derivative_value = float(
            derivative.subs(substitutions)
        )

        # Fehlerbeitrag
        error_term = (
            derivative_value * float(err)
        ) ** 2

        error_sum += error_term

        # Ausgabe des einzelnen Fehlerterms
        print(
            f"Term von {var}: "
            f"{error_term:.10g}".replace(".", ",")
        )

        # --------------------------------------------------
        # Allgemeiner Term
        # --------------------------------------------------

        symbolic_terms.append(
            rf"\left("
            rf"\frac{{\partial {variable_name}}}"
            rf"{{\partial {sp.latex(var)}}}"
            rf"\,\Delta {sp.latex(var)}"
            rf"\right)^2"
        )

        # --------------------------------------------------
        # Term mit symbolisch berechneter Ableitung
        # --------------------------------------------------

        rounded_derivative = derivative.xreplace({
            number: sp.Float(round(float(number), 2))
            for number in derivative.atoms(sp.Float)
        })
        derivative_latex = sp.latex(rounded_derivative)

        derivative_terms.append(
            rf"\left("
            rf"{derivative_latex}"
            rf"\,\Delta {sp.latex(var)}"
            rf"\right)^2"
        )

    # Gesamter Fehler
    error_value = error_sum ** 0.5

    # --------------------------------------------------
    # 3. Allgemeine Gaußsche Formel
    # --------------------------------------------------

    symbolic_formula = (
        rf"\Delta {variable_name} = "
        rf"\sqrt{{"
        + "+".join(symbolic_terms)
        + "}}"
    )

    # --------------------------------------------------
    # 4. Formel mit symbolisch berechneten Ableitungen
    # --------------------------------------------------

    derivative_formula = (
        rf"\sqrt{{"
        + "+".join(derivative_terms)
        + "}}"
    )

    # Gesamte LaTeX-Gleichung
    latex_formula = (
        rf"\begin{{equation*}}"
        rf"{symbolic_formula}"
        rf" = "
        rf"{derivative_formula}"
        rf"\end{{equation*}}"
    )

    # --------------------------------------------------
    # 5. Numerisches Ergebnis mit ±
    # --------------------------------------------------

    value_latex = f"{f_value:.2f}".replace(".", ",")
    error_latex = f"{error_value:.2f}".replace(".", ",")

    latex_result = (
        rf"{variable_name} = "
        rf"({value_latex} \pm {error_latex})"
        rf"{unit}"
    )

    # --------------------------------------------------
    # Ausgabe
    # --------------------------------------------------

    print()
    print("1. Wert:")
    print(f"{f_value:.2f}".replace(".", ","))

    print()
    print("2. Fehler:")
    print(f"{error_value:.2f}".replace(".", ","))

    print()
    print("3. LaTeX-Formel:")
    print(latex_formula)

    print()
    print("4. LaTeX-Ergebnis:")
    print(latex_result)

    # --------------------------------------------------
    # Rückgabe
    # --------------------------------------------------

    if returnLaTex:
        return (
            f_value,
            error_value,
            latex_formula,
            latex_result
        )

    return f_value, error_value

def zTest(Bestwert:float, Literaturwert:float, Standardunsicherheit:float, Einheit:str="", LaTexAusgabe=False):
    """
    Berechnet den z-Wert für die gegebenen Werte.
    """
    z = abs((Bestwert - Literaturwert) / Standardunsicherheit)

    if LaTexAusgabe:
        if z > 2:
            print(f"Da $z = \\left|\\frac{{\hat x-y}}{{\Delta x}}\\right| = \\left|\\frac{{ {Bestwert:.2f} {Einheit} - {Literaturwert:.2f} {Einheit}}}{{{Standardunsicherheit:.2f} {Einheit} }} \\right| = {z:.2f} > 2$, weicht unser Ergebnis Signifikant vom Literaturwert ab.")
        else: print(f"Da $z = \\left|\\frac{{\hat x-y}}{{\Delta x}}\\right| = \\left|\\frac{{ {Bestwert:.2f} {Einheit} - {Literaturwert:.2f} {Einheit}}}{{{Standardunsicherheit:.2f} {Einheit}}}\\right| = {z:.2f} <= 2$, ist unser Ergebnis mit dem Literaturwert (${Literaturwert:.2f} {Einheit}$) kompatibel.")
    return z

def plot_Diagramm(x_vals, y_vals, error, Title:str, x_label:str=None, y_label:str=None):
    """Plottet ein Diagramm mit Fehlerbalken, Titel,"""
    fig, ax = plt.subplots(dpi=600)
    ax.errorbar(x_vals, y_vals, error, fmt='.', linewidth=2, capsize=6)
    ax.set_title(Title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()

def Residuendiagramm_manuell(
    x_vals,
    y_vals,
    error,
    Steigung: float,
    Verschiebung: float,
    Steigung_delta: float = 0,
    Verschiebung_delta: float = 0,
    zeige_grenzgeraden: bool = False,
    x_label: str = None,
    y_label: str = None):
    """Erstellt ein Diagramm mit Konfidenzschlauch und ein Residuendiagramm."""

    gerade = (
        Steigung * x_vals
        + Verschiebung
    )

    x_schwerpunkt = np.mean(x_vals)
    y_drehpunkt = Steigung * x_schwerpunkt + Verschiebung
    x_relativ = x_vals - x_schwerpunkt

    g1 = (Steigung - Steigung_delta) * x_relativ + y_drehpunkt
    g2 = Steigung * x_relativ + y_drehpunkt
    g3 = (Steigung + Steigung_delta) * x_relativ + y_drehpunkt

    g2_oben = g2 + Verschiebung_delta
    g2_unten = g2 - Verschiebung_delta

    band_oben = np.maximum.reduce([
        g1,
        g2_oben,
        g3
    ])
    band_unten = np.minimum.reduce([
        g1,
        g2_unten,
        g3
    ])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.errorbar(
        x_vals,
        y_vals,
        yerr=error,
        fmt=".",
        linewidth=1,
        capsize=4,
        markersize=4
    )

    ax.plot(x_vals, gerade, linewidth=1, color="orange")

    if zeige_grenzgeraden:
        for grenze in (g1, g2_oben, g3, g1, g2_unten, g3):
            ax.plot(x_vals, grenze, "--", linewidth=0.8)

    ax.fill_between(
        x_vals,
        band_unten,
        band_oben,
        alpha=0.2
    )

    # ax.set_title("Ausgleichsgerade")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    plt.show()

    residuen = y_vals - gerade

    residuen_band_unten = band_unten - gerade
    residuen_band_oben = band_oben - gerade

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.plot(x_vals, residuen, ".")

    ax.fill_between(
        x_vals,
        residuen_band_unten,
        residuen_band_oben,
        alpha=0.2
    )

    ax.hlines(
        0,
        x_vals[0],
        x_vals[-1],
        color="orange",
        linewidth=1
    )

    # ax.set_title("Residuen-Diagramm")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    plt.show()


if __name__ == "__main__":
    # anwendung von lade_versuch
    daten, parameter = lade_versuch(
        "reversionspendel.xlsx"
    )
    s = daten["Länge s [m]"].to_numpy()
    messzeiten = daten[
        ["T1 [s]", "T2 [s]", "T3 [s]", "T4 [s]", "T5 [s]"]
    ].to_numpy()

    # Fehlerfortpflanzung 
    print("Fehler Gauss Test:")
    h, k, d1, d2 = sp.symbols('h k d_1 d_2')
    V = h*k*(d1+d2)/2
    V1 = gauss_error_values(
        V,
        [h, k, d1, d2],
        [5.07, 23.04, 34.4, 34.95],
        [0.005, 0.005, 0.05, 0.05],
        returnLaTex=True
    )
