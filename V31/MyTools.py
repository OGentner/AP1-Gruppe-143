import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from decimal import Decimal, ROUND_HALF_UP, localcontext

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

def generate_Table(Tabelle:pd.DataFrame, captionIN:str, labelIN:str):
    """Generiert eine LaTeX-Tabelle aus einem Pandas DataFrame."""
    latex = Tabelle.to_latex(
        index=False,
        bold_rows=False,
        escape=False,
        float_format="{:.2e}".format,
        decimal=',',
        column_format=("l"*(len(Tabelle.columns.to_list()))), # L eft, C enter, R ight
        header=Tabelle.columns.to_list(),
        caption=captionIN,
        label="tab:"+labelIN,
        position="H"
    )

    for i in range(-12, 13):
        if i < 0:
            old_str = f"e{i:03d}"
            new_str = f"$\cdot 10^{{{i}}}$"
        elif i == 0: 
            old_str = "e+00"
            new_str = ""
        elif i == 1:
            old_str = "e+01"
            new_str = "$\cdot 10$"
        elif i > 1:
            old_str = f"e+{i:02d}"
            new_str = f"$\, \cdot 10^{{{i}}}$"
        latex = latex.replace(old_str, new_str)

    latex = latex.replace(r"\begin{table}[H]", r"\begin{table}[H]"+"\n"+r"\centering")
    latex = latex.replace(r"\toprule", r"\hline")
    latex = latex.replace(r"\midrule", r"\hline")
    latex = latex.replace(r"\bottomrule", r"\hline")


    return (latex)

def gewichteter_Mittelwert(daten, fehler):
    """Berechnet den gewichteten Mittelwert der gegebenen Arrays.
    daten = np.array 
    fehler = np.array
    returns [Mittelwert, Fehler]"""
    
    w = 1/(fehler)**2
    omega_bar_w = (np.sum(w*daten)) / (np.sum(w))
    del_omega_bar_w = 1 / np.sqrt(np.sum(w))
    return [omega_bar_w, del_omega_bar_w]

def gauss_error_values(
    f,
    variables,
    values,
    errors,
    eqLabel=None,
    variable_name="f",
    unit="",
    returnLaTex=False,
    *,
    correlation_matrix=None,
    print_output=False
):
    """Berechnet die Gaußsche Fehlerfortpflanzung erster Ordnung.

    ``variables``, ``values`` und ``errors`` müssen gleich lang und in derselben
    Reihenfolge angegeben werden. Ohne ``correlation_matrix`` werden unabhängige
    Eingangsgrößen angenommen.

    Die numerischen Rückgabewerte sind ungerundet. Die LaTeX-Ausgabe rundet die
    Unsicherheit auf eine signifikante Stelle (bei führender 1 oder 2 auf zwei)
    und den Wert auf dieselbe Nachkommastelle.
    """
    expression = sp.sympify(f)
    variables = list(variables)
    values = list(values)
    errors = list(errors)
    count = len(variables)

    if len(values) != count or len(errors) != count:
        raise ValueError(
            "variables, values, and errors must have the same length "
            f"(got {count}, {len(values)}, and {len(errors)})"
        )
    if any(not isinstance(variable, sp.Symbol) for variable in variables):
        raise TypeError("Every entry in variables must be a SymPy Symbol")
    if len(set(variables)) != count:
        raise ValueError("variables must not contain duplicate symbols")
    if eqLabel is not None:
        if not isinstance(eqLabel, str) or not eqLabel.strip():
            raise ValueError("eqLabel must be a non-empty string or None")
        if not all(
            character.isalnum() or character in ":_.-"
            for character in eqLabel
        ):
            raise ValueError("eqLabel may contain only letters, digits, ':', '_', '.', or '-'")

    unprovided_symbols = expression.free_symbols.difference(variables)
    if unprovided_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in unprovided_symbols))
        raise ValueError(f"f contains symbols missing from variables: {names}")

    def finite_real(value, description):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{description} must evaluate to a real number") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"{description} must be finite")
        return numeric_value

    numeric_values = [
        finite_real(value, f"values[{index}]")
        for index, value in enumerate(values)
    ]
    numeric_errors = [
        finite_real(error, f"errors[{index}]")
        for index, error in enumerate(errors)
    ]
    if any(error < 0 for error in numeric_errors):
        raise ValueError("Uncertainties in errors must be non-negative")

    substitutions = dict(zip(variables, numeric_values))
    f_value = finite_real(expression.subs(substitutions), "f")

    derivatives = [sp.diff(expression, variable) for variable in variables]
    gradient = np.array([
        finite_real(derivative.subs(substitutions), f"Derivative for {variable}")
        for variable, derivative in zip(variables, derivatives)
    ])
    contributions = np.abs(gradient * np.array(numeric_errors))
    if not np.all(np.isfinite(contributions)):
        raise ValueError("The propagated uncertainty contributions must be finite")

    if correlation_matrix is None:
        error_value = float(np.linalg.norm(contributions))
    else:
        correlations = np.asarray(correlation_matrix, dtype=float)
        expected_shape = (count, count)
        if correlations.shape != expected_shape:
            raise ValueError(
                "correlation_matrix must have shape "
                f"{expected_shape}, got {correlations.shape}"
            )
        if not np.all(np.isfinite(correlations)):
            raise ValueError("correlation_matrix must contain only finite values")
        if not np.allclose(correlations, correlations.T, rtol=0, atol=1e-12):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(correlations), 1, rtol=0, atol=1e-12):
            raise ValueError("The diagonal of correlation_matrix must contain ones")
        if np.any(np.abs(correlations) > 1 + 1e-12):
            raise ValueError("Correlation coefficients must be between -1 and 1")
        if np.linalg.eigvalsh(correlations).min(initial=0) < -1e-12:
            raise ValueError("correlation_matrix must be positive semidefinite")

        covariance = np.outer(numeric_errors, numeric_errors) * correlations
        if not np.all(np.isfinite(covariance)):
            raise ValueError("The covariance values must be finite")
        variance = float(gradient @ covariance @ gradient)
        if not np.isfinite(variance):
            raise ValueError("The propagated variance must be finite")
        variance_scale = float(np.sum(
            np.abs(np.outer(gradient, gradient) * covariance)
        ))
        tolerance = 1e-12 * variance_scale + np.finfo(float).tiny
        if variance < -tolerance:
            raise ValueError("The propagated variance is negative")
        error_value = float(np.sqrt(max(variance, 0.0)))

    symbolic_terms = []
    derivative_terms = []
    for variable, derivative in zip(variables, derivatives):
        variable_latex = sp.latex(variable)
        symbolic_terms.append(
            rf"\left("
            rf"\frac{{\partial {variable_name}}}{{\partial {variable_latex}}}"
            rf"\,\Delta {variable_latex}"
            rf"\right)^2"
        )
        derivative_terms.append(
            rf"\left("
            rf"\left({sp.latex(derivative)}\right)"
            rf"\,\Delta {variable_latex}"
            rf"\right)^2"
        )

    if not variables:
        variance_formula = "0"
        derivative_formula = "0"
    elif correlation_matrix is None:
        variance_formula = r"\sqrt{" + "+".join(symbolic_terms) + "}"
        derivative_formula = r"\sqrt{" + "+".join(derivative_terms) + "}"
    else:
        variance_formula = (
            rf"\sqrt{{\nabla {variable_name}^{{\mathsf T}}"
            rf"\,\Sigma\,\nabla {variable_name}}}"
        )
        derivative_formula = (
            r"\sqrt{\sum_{i=1}^{n}\sum_{j=1}^{n}"
            rf"\frac{{\partial {variable_name}}}{{\partial x_i}}"
            rf"\frac{{\partial {variable_name}}}{{\partial x_j}}"
            r"\,\operatorname{Cov}(x_i,x_j)}"
        )

    formula_body = (
        rf"\Delta {variable_name} = {variance_formula}"
        rf" = {derivative_formula}"
    )
    if eqLabel is None:
        latex_formula = rf"\begin{{equation*}}{formula_body}\end{{equation*}}"
    else:
        latex_formula = (
            rf"\begin{{equation}}\label{{eq:{eqLabel}}}"
            rf"{formula_body}\end{{equation}}"
        )

    if error_value == 0:
        value_text = f"{f_value:.6g}"
        error_text = "0"
    else:
        error_decimal = Decimal(str(error_value))
        exponent = error_decimal.adjusted()
        leading_digit = int(error_decimal.scaleb(-exponent))
        significant_digits = 2 if leading_digit in (1, 2) else 1
        decimal_places = significant_digits - 1 - exponent
        quantum = Decimal(1).scaleb(-decimal_places)
        value_decimal = Decimal(str(f_value))
        precision = max(
            28,
            error_decimal.adjusted() + decimal_places + 3,
            value_decimal.adjusted() + decimal_places + 3
        )
        with localcontext() as context:
            context.prec = precision
            rounded_error = error_decimal.quantize(
                quantum,
                rounding=ROUND_HALF_UP
            )
            rounded_value = value_decimal.quantize(
                quantum,
                rounding=ROUND_HALF_UP
            )
        value_text = f"{rounded_value:.{max(decimal_places, 0)}f}"
        error_text = f"{rounded_error:.{max(decimal_places, 0)}f}"

    value_latex = value_text.replace(".", "{,}")
    error_latex = error_text.replace(".", "{,}")
    latex_result = (
        rf"{variable_name} = \left({value_latex} \pm {error_latex}\right){unit}"
    )

    if print_output:
        print(f"{variable_name} = {value_text} ± {error_text}{unit}")
        print("1σ-Beiträge je Eingangsgröße:")
        for variable, contribution in zip(variables, contributions):
            print(f"  {variable}: {contribution:.6g}")
        print("LaTeX-Formel:")
        print(latex_formula)
        print("LaTeX-Ergebnis:")
        print(latex_result)

    if returnLaTex:
        return f_value, error_value, latex_formula, latex_result
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
    xerr,
    yerr,
    Steigung: float,
    Verschiebung: float,
    Steigung_delta: float = 0,
    Verschiebung_delta: float = 0,
    zeige_grenzgeraden: bool = False,
    x_label: str = None,
    y_label: str = None):
    """Erstellt ein Diagramm mit Konfidenzschlauch und ein Residuendiagramm."""

    x_plot_min = 0.0
    x_plot_max = float(np.max(x_vals))
    x_plot = np.linspace(x_plot_min, x_plot_max, 500)

    gerade = (
        Steigung * x_plot
        + Verschiebung
    )

    x_schwerpunkt = np.mean(x_vals)
    y_drehpunkt = Steigung * x_schwerpunkt + Verschiebung
    x_relativ = x_plot - x_schwerpunkt

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
        xerr=xerr,
        yerr=yerr,
        fmt=".",
        linewidth=1,
        capsize=4,
        markersize=4
    )

    ax.plot(x_plot, gerade, linewidth=1, color="orange")

    if zeige_grenzgeraden:
        for grenze in (g1, g2_oben, g3, g1, g2_unten, g3):
            ax.plot(x_plot, grenze, "--", linewidth=0.8)

    ax.fill_between(
        x_plot,
        band_unten,
        band_oben,
        alpha=0.2
    )

    # ax.set_title("Ausgleichsgerade")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    plt.show()

    residuen = y_vals - (Steigung * x_vals + Verschiebung)

    residuen_band_unten = band_unten - gerade
    residuen_band_oben = band_oben - gerade

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.plot(x_vals, residuen, ".")

    ax.fill_between(
        x_plot,
        residuen_band_unten,
        residuen_band_oben,
        alpha=0.2
    )

    ax.hlines(
        0,
        x_plot_min,
        x_plot_max,
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
        variable_name="V",
        returnLaTex=True,
        print_output=True
    )
