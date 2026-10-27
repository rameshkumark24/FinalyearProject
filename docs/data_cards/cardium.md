# Data card — CARDIUM (prenatal ultrasound images and clinical data)

**What it is.** Prenatal ultrasound images and one clinical record per patient, with a binary
CHD label, from the BCV group of Universidad de los Andes (Colombia): Vega D, Ceballos HV, Vera JS,
et al. *CARDIUM: Congenital Anomaly Recognition with Diagnostic Images and Unified Medical
records.* ICCV 2025 Workshops (CVAMD). arXiv:2510.15208. Code and access:
github.com/BCV-Uniandes/Cardium.

**Access and terms.** Released through the authors' request form, which checks the requester's
institutional affiliation; dataset licence CC BY-NC 4.0, academic and research use; code Apache
2.0 (repository README and licence files, read on 27 Oct 2026). On the development machine from
24 Oct 2026, in `data/cardium_clinical/` (git-ignored). The authors ask users to cite the paper.

**What may be published.** Aggregate results only (as for Heartbeat): images, index files and
per-patient outputs stay out of the public repository (`tests/test_privacy.py`).

| Item | Value | Where it was checked |
|---|---|---|
| Images | 6,558 PNG in `CARDIUM_dataset/` | files |
| Patients | 1,103 with images (74 with CHD); the clinical table has 1,104 records | files; clinical JSON |
| Labels | the image folders and the clinical table agree for all 1,103 patients | `make_echo_index_cardium.py` |
| Views | not named; some images are colour Doppler or carry colour overlays: 1,224 images have more than 1% strongly coloured pixels; 1,023 patients keep at least one image without | `make_echo_index_cardium.py` |
| Partition | the authors' 3-fold cross-validation by patient (`cardium_images/fold_k/{train,test}`): about 25 patients with CHD and 343 without in each test fold | folders |
| Clinical variables | 29 (blood counts, maternal age, gestational week of each scan, BMI, obstetric and medical history, CHD) | clinical JSON |
| Gestational weeks per scan | listed per patient, but the lists do not line up with the images (equal length for 158 of 1,102 patients), so a week cannot be given to each image | clinical JSON; files |
| Released models | image encoder, tabular encoder and the combined model, one checkpoint per fold | checkpoint files; dataset README |
| Reported result | F1 79.8 ± 4.8% for the combined model | repository README |

**Known problems.**
- The views are not labelled and colour Doppler stills are mixed in; the cleaning step blanks
  coloured pixels, and a grey-only subset is analysed as well.
- **It shares many fetuses with Heartbeat:** 496 of its 1,103 patients share two or more
  cleaned scans with a Heartbeat patient, and 180 more patient pairs share one (run
  `wp14b_echo_overlap_scan`; whole frames never match because the two store frames of different sizes). It is not an
  independent second external dataset.
- **Repeated images with conflicting labels:** 231 identical images appear under
  34 different patients, and 53 of them under patients with different labels.
- The clinical table describes a referral population; its rates do not transfer to screening.

**Index files** (`python scripts/make_echo_index_cardium.py`, written to `data/indexes/`): all
images; grey-only images.
