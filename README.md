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

Gli ID e lo snapshot della selezione sono salvati in `selections/pilot_diverse/`,
separatamente dagli output rigenerabili. Aggiungi questi due JSON a Git dopo la
revisione; la scansione dei candidati resta in `neurovault_test/outputs/pilot_diverse/`.
Se `selected_ids.json` esiste già, lo script si ferma: per riutilizzare gli ID
esistenti passa direttamente al download.

### 2. Download e controlli preliminari

`build_diverse_pilot_v1.py` scarica le mappe selezionate e le annotazioni
Neurosynth usando Nilearn. I dati originali sono conservati in
`neurovault_test/data/pilot_diverse/<SHA256-degli-ID>/`.

I risultati principali in `neurovault_test/outputs/pilot_diverse/` sono:

- `manifest.csv`: metadati, percorsi dei file ed esito dei controlli.
- `basic_pass_ids.json`: ID che superano i controlli preliminari.
- `image_term_weights.csv`: matrice delle annotazioni con ID e nomi dei termini.
- `metadata.json`: informazioni sul dataset e sull'esecuzione.

La selezione è obbligatoria: il builder non avvia ricerche generiche se manca.
Prima di scrivere manifest e annotazioni verifica gli ID esatti, l'assenza di
duplicati, il numero di file e la corrispondenza collezione/contrasto con lo
snapshot. Conserva l'ordine restituito da Nilearn anche nelle righe dei pesi.
Il builder non modifica mai la selezione.

### Recuperare il pilota precedente

Prima di generare una nuova selezione, se sul tuo computer è ancora disponibile
`outputs/pilot_diverse/selection_metadata.json`, puoi recuperare gli ID originali:

```bash
python neurovault_test/select_diverse_pilot.py --import-legacy
python neurovault_test/build_diverse_pilot_v1.py
python neurovault_test/prepare_pilot_volumes_v2.py
```

L'importazione usa lo snapshot originale anziché il vecchio elenco ID, controlla
i limiti di diversità e rifiuta di sovrascrivere una selezione esistente. Se lo
snapshot è assente o non valido si interrompe: non può ricostruire da solo gli
ID perduti. I JSON locali non sono inclusi in questa correzione del codice.
La nuova cache è separata da `data/pilot/`, che resta disponibile.

Se esiste già una selezione valida in `selections/`, passa direttamente al builder.
I vecchi risultati preparati non vengono riparati automaticamente: riesegui il
builder e la preparazione dopo aver recuperato/verificato la selezione.

### 3. Preparazione dei volumi

`prepare_pilot_volumes_v2.py` legge il manifest e gli ID accettati, ricampiona
le mappe sul riferimento MNI152 a 3 mm e costruisce una maschera comune.
Allinea inoltre le annotazioni allo stesso ordine delle immagini.

Le esclusioni e le relative motivazioni sono configurate in
`selections/pilot_diverse/exclusions.json`. Il file iniziale elenca gli ID `1753`,
`1757` e `1759` per copertura geometrica insufficiente nel pilota originale.
Sono esclusioni specifiche dell'analisi, da rivalutare se cambia il campione.
La selezione resta completa; la preparazione riporta soltanto gli ID effettivamente
utilizzati e le esclusioni applicate. Un campione diverso può quindi escludere
un numero diverso di immagini.

Prima del ricampionamento, entrambe le versioni della preparazione verificano
selezione, manifest, ID accettati e impronte SHA256 dei file prodotti dal builder.
Input di provenienza assente o diversa causano un errore. Il riepilogo registra
l'impronta della selezione e dei metadati di input.

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

Per riprodurre uno specifico campione, versiona `selected_ids.json`,
`selection_metadata.json` ed `exclusions.json` sotto `selections/pilot_diverse/`,
e conserva le versioni delle dipendenze. Gli output grandi restano esclusi da Git.
Il seed rende stabile la selezione a parità di candidati; il contenuto dell'API
può cambiare nel tempo. I manifest contengono percorsi assoluti locali e vanno
rigenerati dopo aver spostato il progetto o su un altro computer.

La disponibilità delle immagini e delle annotazioni dipende dai servizi remoti.
Per l'uso e la redistribuzione dei dati, consulta le condizioni e le informazioni
di provenienza delle rispettive collezioni NeuroVault.



## Test di regressione offline

```bash
python -m unittest discover -s tests -v
```

I test richiedono NumPy e usano risposte simulate per i servizi e i volumi.
Controllano selezione obbligatoria, ID mancanti/aggiuntivi/duplicati, diversità,
recupero dello snapshot originale, allineamento annotazioni e cambiamenti dei
file tra download e preparazione. Non effettuano download e non validano la
registrazione anatomica o l'accuratezza del ricampionamento.
