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

    Die Funktion wertet den sympy-Ausdruck ``f`` an den Messwerten aus und
    bestimmt die Standardunsicherheit mit den partiellen Ableitungen:

    * Ohne ``correlation_matrix`` werden unkorrelierte Eingangsgrößen
      angenommen und die Fehlerbeiträge quadratisch addiert.
    * Mit ``correlation_matrix`` wird die vollständige Kovarianzfortpflanzung
      verwendet. Die Matrix muss eine symmetrische, positiv semidefinite
      Korrelationsmatrix mit Einsen auf der Diagonale sein.

    ``variables``, ``values`` und ``errors`` müssen gleich lang und in
    derselben Reihenfolge angegeben werden. ``errors`` enthält nichtnegative
    Standardunsicherheiten (1 sigma); ``values`` und ``errors`` müssen endlich
    und reell sein. Alle Symbole im Ausdruck müssen in ``variables`` enthalten
    sein.

    Args:
        f: Auswertbarer sympy-Ausdruck oder ein von ``sympy.sympify``
            unterstützter Ausdruck.
        variables: Sympy-Symbole, nach denen abgeleitet wird.
        values: Messwerte in derselben Reihenfolge wie ``variables``.
        errors: Standardunsicherheiten zu den Messwerten.
        eqLabel: Optionales LaTeX-Gleichungslabel ohne ``eq:``-Präfix.
        variable_name: Bezeichnung der Ergebnisgröße in der Ausgabe.
        unit: Einheit, die an das formatierte LaTeX-Ergebnis angehängt wird.
        returnLaTex: Gibt zusätzlich die LaTeX-Formel und das formatierte
            LaTeX-Ergebnis zurück.
        correlation_matrix: Optionale Korrelationsmatrix der Eingangsgrößen.
        print_output: Gibt Ergebnis, einzelne 1-sigma-Beiträge und LaTeX aus.

    Returns:
        Ohne ``returnLaTex`` das Tupel ``(Messwert, Standardunsicherheit)``.
        Mit ``returnLaTex`` zusätzlich ``(LaTeX-Formel, LaTeX-Ergebnis)``.
        Die numerischen Rückgabewerte sind ungerundet. Die LaTeX-Ausgabe
        rundet die Unsicherheit auf eine signifikante Stelle (bei führender
        1 oder 2 auf zwei) und den Wert auf dieselbe Nachkommastelle.

    Verwendung:
        Beispiel: Das Volumen eines Quaders wird aus Länge, Breite und Höhe
        berechnet. Messwerte und Unsicherheiten müssen jeweils in derselben
        Reihenfolge wie die Symbole übergeben werden:

        >>> l, b, h = sp.symbols("l b h")
        >>> V, dV, formel, ergebnis = gauss_error_values(
        ...     l * b * h,
        ...     [l, b, h],
        ...     [12.0, 5.0, 2.0],
        ...     [0.1, 0.1, 0.05],
        ...     variable_name="V",
        ...     unit=r"\,\mathrm{cm}^3",
        ...     eqLabel="quader_volumen",
        ...     returnLaTex=True,
        ... )
        >>> print(f"V = {V:.2f} cm^3, Standardunsicherheit = {dV:.2f} cm^3")
        V = 120.00 cm^3, Standardunsicherheit = 3.97 cm^3

        ``V`` und ``dV`` sind die ungerundeten numerischen Ergebnisse.
        ``formel`` enthält die allgemeine Gaußsche Fehlerfortpflanzung und
        ``ergebnis`` das für ein Protokoll formatierte LaTeX-Ergebnis. Die
        Angabe von ``unit`` formatiert nur die Ausgabe; die Funktion prüft
        keine Einheiten. Daher müssen alle Eingangsgrößen und Unsicherheiten
        in zueinander passenden Einheiten angegeben sein. Ohne eine
        Korrelationsmatrix behandelt die Funktion die Eingangsgrößen als
        unkorreliert.

    Raises:
        TypeError: Wenn ein Eintrag in ``variables`` kein Sympy-Symbol ist.
        ValueError: Wenn Eingaben unvereinbar, nicht endlich/reell oder eine
            angegebene Korrelationsmatrix ungültig ist.
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


def test_gauss_error_values():
    """Prüft ``gauss_error_values`` unabhängig von Messdateien und Protokollen.

    Die Referenzwerte werden aus bekannten Ableitungen und der Gaußschen
    Fehlerfortpflanzung analytisch bestimmt. Es werden reguläre Fälle,
    korrelierte Eingangsgrößen, Rückgabeformate und erwartete Eingabefehler
    geprüft. Der Test ändert die geprüfte Funktion nicht: Alle Fehlschläge
    werden gesammelt und am Ende gemeinsam als ``AssertionError`` gemeldet.

    Aufruf im Python-Interpreter: ``Tools.test_gauss_error_values()``.
    Ein problemloser Lauf endet ohne Ausnahme und gibt die Anzahl der
    bestandenen Prüfungen aus.
    """
    fehler = []
    bestanden = 0

    def pruefe(name, test):
        nonlocal bestanden
        try:
            test()
        except Exception as error:
            fehler.append(f"{name}: {type(error).__name__}: {error}")
        else:
            bestanden += 1

    def nahe(actual, expected, beschreibung):
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-14):
            raise AssertionError(
                f"{beschreibung}: erwartet {expected!r}, erhalten {actual!r}"
            )

    def erwartet_fehler(error_type, funktion, *args, **kwargs):
        try:
            funktion(*args, **kwargs)
        except error_type:
            return
        except Exception as error:
            raise AssertionError(
                f"erwartet {error_type.__name__}, erhalten "
                f"{type(error).__name__}: {error}"
            ) from error
        raise AssertionError(f"erwartet wurde {error_type.__name__}")

    x, y, z = sp.symbols("x y z")

    def linearer_fall():
        wert, unsicherheit = gauss_error_values(3*x + 2, [x], [4], [0.2])
        nahe(wert, 14, "Linearer Messwert")
        nahe(unsicherheit, 0.6, "Lineare Unsicherheit")

    pruefe("Lineare Funktion und Ableitung", linearer_fall)

    def nichtlinearer_fall():
        wert, unsicherheit = gauss_error_values(x**2, [x], [3], [0.1])
        nahe(wert, 9, "Nichtlinearer Messwert")
        nahe(unsicherheit, 0.6, "Nichtlineare Unsicherheit")

    pruefe("Nichtlineare Funktion", nichtlinearer_fall)

    def unabhaengige_eingangsfehler():
        wert, unsicherheit = gauss_error_values(
            x*y, [x, y], [3, 4], [0.1, 0.2]
        )
        nahe(wert, 12, "Produkt-Messwert")
        nahe(unsicherheit, np.sqrt(0.52), "Quadratische Fehleraddition")

    pruefe("Mehrere unkorrelierte Eingangsgrößen", unabhaengige_eingangsfehler)

    def nullfehler_und_konstante():
        wert, unsicherheit = gauss_error_values(
            x + y, [x, y], [5, 7], [0, 0.2]
        )
        nahe(wert, 12, "Messwert mit Nullunsicherheit")
        nahe(unsicherheit, 0.2, "Unsicherheit bei einem fehlerfreien Eingang")

        konstante, konstante_unsicherheit = gauss_error_values(7, [], [], [])
        nahe(konstante, 7, "Konstanter Ausdruck")
        nahe(konstante_unsicherheit, 0, "Unsicherheit des konstanten Ausdrucks")

    pruefe("Nullunsicherheit und konstante Funktion", nullfehler_und_konstante)

    def positive_korrelation():
        matrix = [[1, 0.5], [0.5, 1]]
        wert, unsicherheit = gauss_error_values(
            x + y, [x, y], [1, 2], [0.1, 0.2],
            correlation_matrix=matrix,
        )
        nahe(wert, 3, "Messwert mit positiver Korrelation")
        nahe(unsicherheit, np.sqrt(0.07), "Positive Kovarianz")

    pruefe("Positive Korrelation", positive_korrelation)

    def negative_korrelation():
        matrix = [[1, -0.5], [-0.5, 1]]
        _, unsicherheit = gauss_error_values(
            x + y, [x, y], [1, 2], [0.1, 0.2],
            correlation_matrix=matrix,
        )
        nahe(unsicherheit, np.sqrt(0.03), "Negative Kovarianz")

    pruefe("Negative Korrelation", negative_korrelation)

    def voll_korrelierte_eingaenge():
        matrix = [[1, 1], [1, 1]]
        _, unsicherheit = gauss_error_values(
            x + y, [x, y], [1, 2], [0.1, 0.2],
            correlation_matrix=matrix,
        )
        nahe(unsicherheit, 0.3, "Voll korrelierte Eingangsgrößen")

    pruefe("Positiv semidefinite Korrelationsmatrix", voll_korrelierte_eingaenge)

    def latex_rueckgabe():
        rueckgabe = gauss_error_values(
            x, [x], [12.345], [0.678],
            eqLabel="test_label",
            variable_name="R",
            unit=r"\,\mathrm{m}",
            returnLaTex=True,
        )
        if len(rueckgabe) != 4:
            raise AssertionError(f"4 Rückgabewerte erwartet, erhalten: {len(rueckgabe)}")
        wert, unsicherheit, formel, ergebnis = rueckgabe
        nahe(wert, 12.345, "Ungerundeter LaTeX-Messwert")
        nahe(unsicherheit, 0.678, "Ungerundete LaTeX-Unsicherheit")
        if r"\label{eq:test_label}" not in formel:
            raise AssertionError(f"Gleichungslabel fehlt: {formel}")
        if "12{,}3" not in ergebnis or "0{,}7" not in ergebnis:
            raise AssertionError(f"LaTeX-Rundung unerwartet: {ergebnis}")
        if r"\,\mathrm{m}" not in ergebnis:
            raise AssertionError(f"Einheit fehlt in LaTeX-Ausgabe: {ergebnis}")

    pruefe("LaTeX-Rückgabe, Label, Einheit und Rundung", latex_rueckgabe)

    pruefe(
        "Ungleiche Listenlängen werden abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x, [x], [1, 2], [0.1]
        ),
    )
    pruefe(
        "Doppelte Variablen werden abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x, [x, x], [1, 2], [0.1, 0.2]
        ),
    )
    pruefe(
        "Nicht angegebene Symbole werden abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x + y, [x], [1], [0.1]
        ),
    )
    pruefe(
        "Nicht-Sympy-Variablen werden abgewiesen",
        lambda: erwartet_fehler(
            TypeError, gauss_error_values, x, ["x"], [1], [0.1]
        ),
    )
    pruefe(
        "Negative Unsicherheiten werden abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x, [x], [1], [-0.1]
        ),
    )
    pruefe(
        "Nicht-endliche Messwerte werden abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x, [x], [np.inf], [0.1]
        ),
    )
    pruefe(
        "Falsche Korrelationsmatrix-Dimension wird abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x + y, [x, y], [1, 2], [0.1, 0.2],
            correlation_matrix=[[1]],
        ),
    )
    pruefe(
        "Asymmetrische Korrelationsmatrix wird abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x + y, [x, y], [1, 2], [0.1, 0.2],
            correlation_matrix=[[1, 0.2], [0.3, 1]],
        ),
    )
    pruefe(
        "Nicht positiv semidefinite Korrelationsmatrix wird abgewiesen",
        lambda: erwartet_fehler(
            ValueError, gauss_error_values, x + y + z, [x, y, z],
            [1, 2, 3], [0.1, 0.2, 0.3],
            correlation_matrix=[
                [1, 0.9, 0.9],
                [0.9, 1, -0.9],
                [0.9, -0.9, 1],
            ],
        ),
    )

    print(
        f"gauss_error_values: {bestanden} Prüfungen bestanden, "
        f"{len(fehler)} fehlgeschlagen."
    )
    if fehler:
        details = "\n".join(f"- {eintrag}" for eintrag in fehler)
        raise AssertionError(f"Fehler in gauss_error_values:\n{details}")


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
    test_gauss_error_values()
    
