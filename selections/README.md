# Selezioni versionate

`pilot_diverse/selected_ids.json` e `pilot_diverse/selection_metadata.json`
sono prodotti dal selettore e devono essere aggiunti a Git dopo la revisione.
Non contengono volumi o percorsi assoluti della cache locale.

Il repository non include ID ricostruiti o inventati per il pilota originale:
per recuperarli dallo snapshot locale usare:

```bash
python neurovault_test/select_diverse_pilot.py --import-legacy
```

L'importazione legge `outputs/pilot_diverse/selection_metadata.json`, verifica
ID e limiti di diversità e non si fida del vecchio `selected_ids.json`.
Se lo snapshot originale è assente o non valido, si interrompe.
In alternativa eseguire il selettore senza argomenti per una nuova selezione.

`exclusions.json` registra le esclusioni specifiche dell'analisi e le motivazioni.
Le tre esclusioni iniziali derivano dai controlli del pilota originale; non sono
regole generali di qualità di NeuroVault. La selezione originale resta completa,
mentre `excluded_images.csv` documenta quali esclusioni sono state applicate.
