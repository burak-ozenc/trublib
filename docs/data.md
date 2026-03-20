# Data sources

This file documents every audio source used to train the bundled classifier model, the license of each source, and the resulting usage constraints on the model weights.

---

## License summary

| Layer | License |
|---|---|
| **Source code** | MIT — unrestricted |
| **Trained model weights** | Non-commercial / research / educational use only |

The source code is MIT. The trained model weights are derived from datasets that carry NonCommercial terms (IRMAS, ESC-50, good-sounds). Under those upstream licenses, any artifact trained on those datasets inherits the non-commercial restriction. The code and model are therefore licensed separately.

If you retrain the model exclusively from CC0 / CC BY sources (MUSAN, tinySOL, Medley-solos-DB, Philharmonia, VSCO 2 CE, your own recordings), the resulting weights are commercially usable — the NC constraint comes entirely from which datasets you include.

---

## Trumpet class — positive examples

### Personal recordings

Original recordings made specifically for this project. No license restrictions.

| Detail | Value |
|---|---|
| Content | Scales, études, long tones, various articulations, muted playing |
| Formats | WAV, recorded at 44.1 kHz, converted to 24 kHz mono |
| License | Original content — unrestricted |

### Freesound packs

Individual Freesound tracks carry their own per-file license (CC0, CC BY, CC BY-NC, etc.). Each pack page lists the license per file. All packs below were used for non-commercial training; verify per-file licenses before commercial use.

| Pack | Freesound ID | Uploader | Content |
|---|---|---|---|
| Chord-Scale Dataset | [24075](https://freesound.org/people/emirdemirel/packs/24075/) | [emirdemirel](https://freesound.org/people/emirdemirel/) | Scale patterns |
| Blown Sounds | [19198](https://freesound.org/people/FullMetalJedi/packs/19198/) | [FullMetalJedi](https://freesound.org/people/FullMetalJedi/) | Mixed blown sounds |
| Trumpet samples VG Trumpet soft | [13522](https://freesound.org/people/Vlad99/packs/13522/) | [Vlad99](https://freesound.org/people/Vlad99/) | Single notes |
| Trumpet | [35283](https://freesound.org/people/KhalDrogo12/packs/35283/) | [KhalDrogo12](https://freesound.org/people/KhalDrogo12/) | Single notes, phrases |
| Trumpet Fanfares From Sensible to Silly | [23284](https://freesound.org/people/joepayne/packs/23284/) | [joepayne](https://freesound.org/people/joepayne/) | Fanfares |
| Pablo_Project trumpet overall quality of single note | [15850](https://freesound.org/people/fuannnakimochi/packs/15850/) | [fuannnakimochi](https://freesound.org/people/fuannnakimochi/) | Single notes |
| Trumpet Sound Pack | [35307](https://freesound.org/people/KhalDrogo12/packs/35307/) | [KhalDrogo12](https://freesound.org/people/KhalDrogo12/) | Phrases |
| VSCO 2 CE — Trumpet sustain | [21050](https://freesound.org/people/sgossner/packs/21050/) | [sgossner](https://freesound.org/people/sgossner/) | Sustain notes |
| VSCO 2 CE — Trumpet vibrato sustain | [21051](https://freesound.org/people/sgossner/packs/21051/) | [sgossner](https://freesound.org/people/sgossner/) | Vibrato sustain |
| VSCO 2 CE — Trumpet staccato | [21048](https://freesound.org/people/sgossner/packs/21048/) | [sgossner](https://freesound.org/people/sgossner/) | Staccato |
| VSCO 2 CE — Trumpet straight mute sustain | [21049](https://freesound.org/people/sgossner/packs/21049/) | [sgossner](https://freesound.org/people/sgossner/) | Muted sustain |
| VSCO 2 CE — Trumpet harmon mute sustain | [21047](https://freesound.org/people/sgossner/packs/21047/) | [sgossner](https://freesound.org/people/sgossner/) | Harmon mute |

### IRMAS

| Detail | Value |
|---|---|
| Authors | Juan J. Bosch, Ferdinand Fuhrmann, Perfecto Herrera |
| Homepage | <https://www.upf.edu/web/mtg/irmas> |
| License | CC BY-NC-SA 3.0 — **non-commercial, no redistribution of raw audio** |
| Used for | Trumpet excerpts from the instrument recognition corpus |
| ⚠ Model constraint | Training with IRMAS data means model weights must be marked non-commercial |

### tinySOL

| Detail | Value |
|---|---|
| Authors | Cella, Ghisi, Lostanlen, Lévy, Fineberg, Maresz |
| Homepage | <https://zenodo.org/records/3685367> |
| License | CC BY 4.0 — commercially usable with attribution |
| Used for | Orchestral trumpet single notes across dynamic levels |

### Medley-solos-DB

| Detail | Value |
|---|---|
| Authors | Lostanlen, Cella, Bittner, Essid |
| Homepage | <https://zenodo.org/records/3464194> |
| License | CC BY 4.0 — commercially usable with attribution |
| Used for | Trumpet excerpts from the instrument identification corpus |

### Philharmonia Orchestra sound samples

| Detail | Value |
|---|---|
| Organization | Philharmonia Orchestra |
| Homepage | <https://philharmonia.co.uk/resources/sound-samples/> |
| License | Free to use including commercially; raw samples must not be redistributed or sold as samples |
| Used for | Professional-quality trumpet single notes, various articulations |
| ⚠ Note | Do not redistribute the raw audio files |

---

## Non-trumpet class — negative examples

### MUSAN

The primary source for speech and noise negative examples.

| Detail | Value |
|---|---|
| Authors | David Snyder, Guoguo Chen, Daniel Povey |
| Homepage | <https://www.openslr.org/17/> |
| License | CC BY 4.0 — commercially usable with attribution |
| Used for | Speech (multi-language), music, environmental noise |
| Citation | Snyder et al., "MUSAN: A Music, Speech, and Noise Corpus", 2015 |

### ESC-50

| Detail | Value |
|---|---|
| Author | Karol J. Piczak |
| Homepage | <https://github.com/karolpiczak/ESC-50> |
| License | CC BY-NC for the full dataset; CC BY for ESC-10 subset |
| Used for | Environmental and ambient noise |
| ⚠ Model constraint | Training with full ESC-50 means model weights must be marked non-commercial |

### good-sounds

| Detail | Value |
|---|---|
| Authors | Romani Picas, Parra Rodriguez, Bandiera, Dabiri, Serra |
| Homepage | <https://www.upf.edu/web/mtg/good-sounds> |
| License | CC BY-NC 4.0 — **non-commercial** |
| Used for | Non-trumpet instrument recordings (trombone, violin, saxophone, tuba, french horn) |
| ⚠ Model constraint | Training with good-sounds means model weights must be marked non-commercial |

### Breathing recordings

| Detail | Value |
|---|---|
| Source | Collected recordings of player breathing, breath attacks, and room tone |
| License | Original content — unrestricted |
| Used for | Preventing the model from triggering on player breath sounds before a note |

---

## Datasets reviewed but not used

### OrchideaSOL

Reviewed but not included. The audio data is licensed under the Ircam Forum License (not a standard CC license), which requires Ircam Forum membership for download. Homepage: <https://zenodo.org/records/3686252>.

---

## What the NC constraint means in practice

The model bundled with trublib was trained on data that includes IRMAS, ESC-50 (full), and good-sounds, all of which carry NonCommercial terms. The resulting constraint:

- ✅ Use trublib and the bundled model for personal practice tools, research, education, and non-commercial applications
- ✅ Use the MIT-licensed source code in any project, including commercial ones
- ❌ Do not use the bundled model weights in a commercial product or service

**To produce commercially-usable weights:** retrain using only CC0 / CC BY sources: MUSAN, tinySOL, Medley-solos-DB, Philharmonia, VSCO 2 CE (sgossner packs), and your own recordings. Exclude IRMAS, ESC-50 (use ESC-10 subset instead), and good-sounds.

---

## Attribution notice

If you publish work that uses the trublib model or trains from these sources, include the following attributions:

```
Training data includes:
- MUSAN (Snyder et al., CC BY 4.0) — https://www.openslr.org/17/
- IRMAS (Bosch et al., CC BY-NC-SA 3.0) — https://www.upf.edu/web/mtg/irmas
- ESC-50 (Piczak, CC BY-NC) — https://github.com/karolpiczak/ESC-50
- good-sounds (Romani Picas et al., CC BY-NC 4.0) — https://www.upf.edu/web/mtg/good-sounds
- tinySOL (Cella et al., CC BY 4.0) — https://zenodo.org/records/3685367
- Medley-solos-DB (Lostanlen et al., CC BY 4.0) — https://zenodo.org/records/3464194
- Philharmonia Orchestra sound samples — https://philharmonia.co.uk/resources/sound-samples/
- VSCO 2 Community Edition (sgossner, CC0) — https://freesound.org/people/sgossner/
- Freesound packs (various uploaders, see docs/data.md for per-pack details)
```