# BESS Arbitrage Optimierung

Dieses Projekt analysiert die techno-ökonomische Arbitrage-Strategie von Batteriespeichersystemen (BESS) am deutschen Day-Ahead-Strommarkt. Es verwendet ein quadratisches Optimierungsmodell, um die Lade- und Entladeentscheidungen eines BESS zu simulieren und die erzielbaren Erlöse unter Berücksichtigung von physikalischen und ökonomischen Parametern zu bewerten.

## Kernkomponenten

Das Projekt besteht aus drei Hauptskripten:

1.  **`data_fetch.py`**: Dieses Skript lädt historische Marktdaten (Day-Ahead-Preise, Last, Erzeugung aus Erneuerbaren) von der ENTSO-E Transparenzplattform für den Zeitraum 2019-2025. Die aufbereiteten Daten werden in `market_data_2019_2025.csv` gespeichert.

2.  **`Update.py`**: Das Kernstück des Projekts. Es führt die BESS-Simulation auf Basis der geladenen Marktdaten durch. Mittels eines quadratischen Optimierungsmodells (OSQP) werden die Lade-/Entladezyklen optimiert, um den Arbitrage-Erlös zu maximieren. Das Modell berücksichtigt dabei:
    *   Wirkungsgrade (Laden/Entladen)
    *   C-Rate (Leistung im Verhältnis zur Kapazität)
    *   Degradationskosten (als Hurdle-Rate)
    *   Preiseffekt (Einfluss großer Speichersysteme auf den Marktpreis)
    
    Die Ergebnisse werden in `simulation_decisions_with_pt.csv` (stündliche Entscheidungen) und `arbitrage_summary_with_pt.csv` (zusammengefasste Erlöse) gespeichert.

3.  **`presentation_analysis.py`**: Dieses Skript dient der Analyse und Visualisierung der Simulationsergebnisse. Es generiert eine Reihe von Plots zur Auswertung der Performance, Sensitivität und Wirtschaftlichkeit der BESS-Strategien und speichert diese als `.png`-Dateien im Hauptverzeichnis.

## Setup

### 1. Voraussetzungen
*   Python 3.8 oder höher
*   Ein persönlicher API-Schlüssel von der [ENTSO-E Transparenzplattform](https://transparency.entsoe.eu/).

### 2. Installation
1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd BESS_Optimization
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren:**
    *   Auf macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   Auf Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. API-Schlüssel konfigurieren
Um Marktdaten abrufen zu können, benötigen Sie einen API-Schlüssel von ENTSO-E.

1.  Erstellen Sie eine Kopie der Vorlagedatei `.env.example` und nennen Sie sie `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Öffnen Sie die neu erstellte `.env`-Datei mit einem Texteditor.
3.  Ersetzen Sie `YOUR_API_KEY_HERE` mit Ihrem persönlichen ENTSO-E API-Schlüssel. Die Datei sollte danach so aussehen:
    ```
    ENTSOE_API_KEY="dein-persönlicher-api-schlüssel"
    ```
Die `.env`-Datei wird durch `.gitignore` ignoriert und somit niemals im Git-Repository gespeichert.

## Workflow & Nutzung

Die Skripte sollten in der folgenden Reihenfolge ausgeführt werden:

1.  **Marktdaten herunterladen:**
    Führen Sie dieses Skript einmalig aus, um die benötigten Daten von ENTSO-E zu laden.
    ```bash
    python data_fetch.py
    ```
    *Output*: `market_data_2019_2025.csv`

2.  **Simulation durchführen:**
    Starten Sie die Arbitrage-Simulation für die verschiedenen Szenarien. Dieses Skript nutzt alle verfügbaren CPU-Kerne für eine parallele Berechnung.
    ```bash
    python Update.py
    ```
    *Outputs*: `arbitrage_summary_with_pt.csv`, `simulation_decisions_with_pt.csv`

3.  **Analyse und Plots generieren:**
    Nach Abschluss der Simulation können Sie dieses Skript ausführen, um alle Grafiken für die Ergebnispräsentation zu erstellen.
    ```bash
    python presentation_analysis.py
    ```
    *Outputs*: Diverse `plot_*.png` Dateien.
