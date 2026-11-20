# Brain Basis Imaging

Progetto sperimentale in Python per costruire un dataset pilota di mappe
statistiche fMRI da NeuroVault, recuperare le annotazioni Neurosynth e preparare
matrici immagini–voxel e immagini–termini per analisi successive.

Il codice include controlli preliminari sui volumi, ricampionamento su una griglia
MNI comune, immagini per il controllo visivo e una prova di proiezione corticale.

## Installazione

Dalla cartella principale del repository, crea un ambiente virtuale e installa
le librerie usate dagli script:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas nibabel nilearn matplotlib requests urllib3 neuromaps
```

I comandi di attivazione sono per Linux/macOS. Le versioni delle dipendenze non
sono ancora fissate in un file dedicato.

I download richiedono una connessione a Internet. Gli script salvano dati e
risultati in percorsi relativi alla propria posizione, dentro `neurovault_test/`.
I parametri si modificano direttamente negli script: al momento non esiste
un'interfaccia a riga di comando con opzioni.

## Dataset pilota: flusso principale

Esegui questi comandi nell'ordine indicato, dalla cartella principale:

```bash
python neurovault_test/select_diverse_pilot.py
python neurovault_test/build_diverse_pilot_v1.py
python neurovault_test/prepare_pilot_volumes_v2.py
```

### 1. Selezione delle mappe

`select_diverse_pilot.py` consulta i metadati dell'API NeuroVault e cerca
40 mappe Z fMRI-BOLD, non sogliate, dichiarate valide e in spazio MNI.
La selezione usa un seed pari a `42`, con un massimo di 5 mappe per collezione
e 2 per coppia collezione–contrasto, per limitare la concentrazione del campione.

Gli ID e i metadati sono salvati in `neurovault_test/outputs/pilot_diverse/`.
Se `selected_ids.json` esiste già, lo script si ferma: per riutilizzare gli ID
esistenti passa direttamente al download.

### 2. Download e controlli preliminari

`build_diverse_pilot_v1.py` scarica le mappe selezionate e le annotazioni
Neurosynth usando Nilearn. I dati originali sono conservati in
`neurovault_test/data/pilot/`.

I risultati principali in `neurovault_test/outputs/pilot_diverse/` sono:

- `selected_ids.json`: ID delle immagini scaricate.
- `manifest.csv`: metadati, percorsi dei file ed esito dei controlli.
- `basic_pass_ids.json`: ID che superano i controlli preliminari.
- `image_term_weights.csv`: matrice delle annotazioni con ID e nomi dei termini.
- `metadata.json`: informazioni sul dataset e sull'esecuzione.

Esegui la selezione prima del download per applicare i criteri di diversità:
in assenza di `selected_ids.json`, il builder effettua una ricerca generica
di mappe fMRI-BOLD di tipo Z.

### 3. Preparazione dei volumi

`prepare_pilot_volumes_v2.py` legge il manifest e gli ID accettati, ricampiona
le mappe sul riferimento MNI152 a 3 mm e costruisce una maschera comune.
Allinea inoltre le annotazioni allo stesso ordine delle immagini.

Questa versione esclude esplicitamente gli ID `1753`, `1757` e `1759` per
copertura geometrica insufficiente osservata nel campione corrente. Sono
esclusioni specifiche dell'analisi, da rivalutare se cambia il campione.

I risultati sono salvati in
`neurovault_test/outputs/pilot_diverse/prepared_fullcoverage/`:

| File o cartella | Contenuto |
| --- | --- |
| `X_voxels_raw.npy` | Matrice immagini × voxel comuni, senza normalizzazione |
| `Y_terms.npy` | Matrice immagini × termini, allineata alle righe di X |
| `image_ids.npy` | ID nell'ordine delle righe delle matrici |
| `voxel_indices.npy` | Indici dei voxel nell'ordine delle colonne di X |
| `aligned_image_term_weights.csv` | Annotazioni allineate, con etichette |
| `common_mask.nii.gz` | Maschera comune ai volumi |
| `geometric_coverage_fraction.nii.gz` | Frazione di immagini che coprono ciascun voxel |
| `reference_mni_3mm.nii.gz` | Riferimento anatomico usato per il ricampionamento |
| `reference_brain_mask.nii.gz` | Maschera cerebrale del riferimento |
| `excluded_images.csv` | Immagini escluse e motivazione |
| `preparation_summary.json` | Dimensioni, copertura e riepilogo della preparazione |
| `volumes/` | Volumi ricampionati |
| `figures/` | Figure per il controllo visivo |

La maschera resta provvisoria in attesa di revisione visiva. Il ricampionamento
su una griglia comune non esegue una registrazione anatomica dei volumi.

## Script esplorativi e versioni precedenti

| Script in `neurovault_test/` | Scopo |
| --- | --- |
| `test_neurovault_v1.py` | Scarica le immagini di esempio 32015 e 32016 e produce figure e metadati |
| `test_neurosynth_terms_v1.py` | Verifica il recupero delle annotazioni per le immagini di esempio |
| `check_align_v1.py` | Produce sovrapposizioni delle immagini di esempio al riferimento MNI |
| `project_to_surface_v1.py` | Proietta l'immagine 32015 su fsaverage a densità 10k |
| `build_pilot_dataset_v1.py` | Costruisce un primo pilota senza la selezione diversificata |
| `prepare_pilot_volumes.py` | Prima preparazione volumetrica, con risultati in `pilot_diverse/prepared/` |

Per provare il controllo di allineamento e la proiezione su superficie:

```bash
python neurovault_test/test_neurovault_v1.py
python neurovault_test/check_align_v1.py
python neurovault_test/project_to_surface_v1.py
```

I file con prefisso `test_` sono script esplorativi che possono scaricare dati
e scrivere risultati; non costituiscono una suite di test automatici.

## Dati e riproducibilità

Il repository esclude dati scaricati, output generati, ambienti virtuali e
configurazioni locali tramite `.gitignore`. Queste cartelle vengono ricreate
dagli script.

Per riprodurre uno specifico campione, conserva separatamente
`selected_ids.json`, i metadati della selezione e le versioni delle dipendenze.
Il seed rende stabile la selezione a parità di candidati; il contenuto dell'API
può cambiare nel tempo. I manifest contengono percorsi assoluti locali e vanno
rigenerati dopo aver spostato il progetto o su un altro computer.

La disponibilità delle immagini e delle annotazioni dipende dai servizi remoti.
Per l'uso e la redistribuzione dei dati, consulta le condizioni e le informazioni
di provenienza delle rispettive collezioni NeuroVault.
