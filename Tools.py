import pandas as pd

datei = "reversionspendel.xlsx"

daten = pd.read_excel(
    datei,
    sheet_name="Messdaten"
)

print(daten)
s = daten["Länge s [m]"].to_numpy()

T1 = daten["T1 [s]"].to_numpy()
T2 = daten["T2 [s]"].to_numpy()
T3 = daten["T3 [s]"].to_numpy()
T4 = daten["T4 [s]"].to_numpy()
T5 = daten["T5 [s]"].to_numpy()

print(T1)
messzeiten = daten[
    ["T1 [s]", "T2 [s]", "T3 [s]", "T4 [s]", "T5 [s]"]
].to_numpy()

print(messzeiten)

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
