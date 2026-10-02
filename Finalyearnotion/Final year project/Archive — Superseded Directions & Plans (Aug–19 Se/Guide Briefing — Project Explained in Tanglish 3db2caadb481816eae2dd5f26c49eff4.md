# Guide Briefing — Project Explained in Tanglish

<aside>
🗣️

**Indha page enna?** Namma project oda basic understanding, Tanglish la. Guide kitta explain panna (15 Sep 2026) and Review II ku prepare panna.

Ellaa numbers um indha workspace la already primary source oda irukkura pages la irundhu dhaan: [Review II — Presentation Content (17 Sep 2026)](Review%20II%20%E2%80%94%20Presentation%20Content%20(17%20Sep%202026)%203db2caadb48181f6899fe7de764822c9.md), [Review I — Research Package](Review%20I%20%E2%80%94%20Research%20Package%203d42caadb4818121982ad98ebe1065ba.md), [Experiment Plan — What to Try, in Order](Experiment%20Plan%20%E2%80%94%20What%20to%20Try,%20in%20Order%203d42caadb481819e95bbf1c6159f198f.md). Step-by-step plan: [Roadmap — Steps to the Paper (14 Sep → 12 Oct 2026)](Roadmap%20%E2%80%94%20Steps%20to%20the%20Paper%20(14%20Sep%20%E2%86%92%2012%20Oct%202026%203db2caadb481819e954ef911bed199c2.md).

</aside>

---

# 1. Oru line la project

> **"Phone la edutha ECG printout photo va correct aana ECG signal ah maathurom. Adhu evlo correct nu doctor use panra units la (BPM, ms, mV) measure panrom."**
> 

---

# 2. Problem enna?

- Hospital la ECG machine heartbeat ah **paper la print** pannum. India la neraya clinics la ECG paper form la mattum dhaan irukkum.
- Adha innoruthar kitta anuppa illa software ku kudukka, makkal **phone la photo** edupanga.
- Aana AI models ellam ECG ah **signal (numbers)** ah dhaan expect pannum, photo ah illa.
- So photo va signal ah maathanum. Idhukku peru **ECG digitization**.
- **Main problem:** ippo irukkura methods ellam **scanner la scan panna clean image** la dhaan build and test pannirukanga. Phone photo kuduthaa fail aagudhu.

## Yen phone photo kashtam? (idhu dhaan key concept)

ECG paper la irukka **grid (kattam)** dhaan units kudukkudhu:

| Direction | Standard | Oru chinna square (1 mm) |
| --- | --- | --- |
| Horizontal (time) | 25 mm = 1 second | **40 ms** |
| Vertical (voltage) | 10 mm = 1 mV | **0.1 mV** |
- Scan panna grid ellaa idathulayum same size la irukkum.
- Phone la **angle la** photo edutha, grid oru side periyadhaavum innoru side chinnadhaavum theriyum. Idhu dhaan perspective distortion.
- Grid size ah thappa kanakku panna, ella ms um mV um thappa varum. **Aana waveform shape paaka correct ah dhaan irukkum.** Adhu dhaan danger.

<aside>
💡

**Oru example (purinjukka mattum, measured result illa):** grid scale 10% thappa irundha, 400 ms QT **440 ms** nu varum. Line paaka same, aana number thappu.

</aside>

---

# 3. Idhu real problem nu eppadi prove panrom?

**PhysioNet Challenge 2024** oru world-level competition (Reyna et al., CinC 2024). Adhoda official score table la winner team oda results:

| Image type | SNR (dB) |
| --- | --- |
| Colour scan, clean paper | **+4.930** |
| B&W scan, clean paper | +3.479 |
| Colour scan, deteriorated paper | +0.506 |
| Phone photo, clean paper | **−1.071** |
| Phone photo, stained paper | **−0.723** |
| Phone photo, deteriorated paper | **−1.304** |
| Computer monitor photo | **−1.759** |
- **SNR enna?** Output original signal ku evlo close ah irukku nu dB la solra number. Adhigama irundha nalladhu.
- **Negative SNR na** signal ah vida noise adhigam. Adhavadhu phone photo la winner oda output kooda useless.
- Innoru open-source method (npj Digital Medicine, 2025) **19.65 dB** vaangirukku. Aana adhu **scanned paper la mattum** dhaan.

**Summary:** clinic la nijama circulate aagura phone photo thaan field handle panna kashtapadra case.

---

# 4. Research gap — 5 points

<aside>
🎯

**Strong point:** first 4 gaps naama create pannadhu illa. **Challenge organisers avangaloda paper la eluthirukanga.**

</aside>

1. **SNR clinical measurement ah reflect pannala.** Doctor dB paaka maataanga; QT ms, ST mV dhaan paapanga.
2. **Digitize pannanuma venama nu yaarukkum theriyadhu.** Negative SNR vaangina teams kooda classification la nalla score vaanginaanga.
3. **Phone photo dhaan unsolved case.** Mela irukka table dhaan proof.
4. **Reproducibility illa.** Organisers kooda matha methods oda code ah run panni test panna mudiyala.
5. **Geography.** Public dataset la Germany, USA, Norway data mattum dhaan irukku; **Indian ECG printouts illa**.

---

# 5. Namma solution / contribution

**Strategy:** puthu model build panni 19.65 dB ah beat panna try panna maatom. Andha number funded lab oda, clinical data vechu vandhadhu; andha race la jeikka kashtam. **Innum yaarum number publish pannadha idathula dhaan compete panrom.**

| No. | Contribution | Simple ah | Phase |
| --- | --- | --- | --- |
| **C1** | **Clinical-parameter error metric** (main) | Original signal um digitized signal um NeuroKit2 ku kudupom. Adhu P, QRS, T wave start/end kandupidikkum. Adhula irundhu HR, PR, QRS, QT, ST calculate panni rendukkum difference edupom. Andha difference dhaan error. Adhu SNR oda evlo match aagudhu nu correlation um paapom | Phase I |
| C2 | Per-image-type benchmark | Scan, photo results ah separate ah report panrom. Average panna maatom, yen na average dhaan photo failure ah maraikkum | Phase I |
| C3 | Failure reason | Fail aagradhu line kandupidikkaradhula ah (segmentation) illa grid scale la ah (calibration) nu pirichu paakrom | Phase I → II |
| C4 | Three-arm classification | Photo va direct ah classify pannradhu vs digitize panni classify pannradhu vs rendum sethu | Phase II |
| C5 | Indian printouts | PTB-XL signal ah Indian ECG grid paper la print panni photo edukrom. Patient data illa, so ethics approval problem illa (**Route A**) | Phase II |
| C6 | Code release | Code, scripts, results ellam public | Throughout |

**C1 yen main contribution?**

- GPU, training, puthu data edhuvum venam.
- Organisers ketta question ku direct answer.
- Idhu oru metric, so future la matha researchers use panna cite pannuvanga.

---

# 6. Eppadi panna porom — methodology

## Flow (slide 8 diagram)

**PTB-XL signal → ECG-Image-Kit (scan-like/photo-like images) → Digitizer 5 stages → Alignment → Metrics → Results**

## Digitizer oda 5 stages

1. **Segmentation:** image la edhu ECG line, edhu grid, edhu text nu pirikkum.
2. **Perspective correction:** angle la irukka photo va nera aakkum.
3. **Layout identification:** 12 leads la endha lead enga irukku nu kandupidikkum.
4. **Grid size extraction:** grid square size vechu mm/s, mm/mV scale kandupidikkum. ← **Namma target (diagram la orange)**
5. **Segmentation-to-trace:** line ah numbers (signal) ah maathum.

## Adhukku apram

- **Alignment:** output um original um same length and position ku kondu varanum; illana compare panna mudiyadhu.
- **Signal metrics:** SNR, PCC (shape match), RMSE (mV error).
    - PCC shape ah mattum dhaan paakum. Scale thappa irundhaalum PCC nalla score kaatum; adhu oru trap.
- **Clinical metrics:** NeuroKit2 vechu HR, PR, QRS, QT, ST error. ← **Diagram la green, namma contribution**

## Calibration pulse (important idea)

- ECG paper la **1 mV calibration pulse** print aagirukkum: 10 mm uyaram, 0.2 s agalam. Adhu page mela irukka oru **ruler** maadhiri.
- Grid mangala theriyum podhu, andha pulse vechu scale correct ah kandupidikka mudiyuma nu test panna porom (Experiment A2).

## Experiment rules

- **Oru time la oru factor mattum** maathuvom (crease, resolution, grid colour…).
- Dev set, test set munnadiye freeze pannuvom. Test set **oru thadava mattum** use pannuvom.
- Run pannradhukku munnadiye expected result eludhuvom.
- Fail aana runs um log pannuvom.

## Hypotheses — prove panna porom (illa disprove)

- **H1:** Same digitizer ku photo la SNR scan ah vida kammi.
- **H2:** Photo la main error grid scale la dhaan, line kandupidikkaradhula illa.
- **H3:** SNR um clinical error um weak ah dhaan match aagum.
- **H4 (Phase II):** Photo va direct ah classify pannradhu digitize panni classify pannradhu alavukku illa adhai vida nalla varum.
- **H5 (Phase II):** Indian printouts la accuracy innum kammi aagum.

---

# 7. Dataset & tools

| Dataset | Enna | Yen |
| --- | --- | --- |
| **PTB-XL** | 21,799 records / 18,869 patients, 12-lead, 10 sec, 500 Hz | Original (ground truth) signal |
| **ECG-Image-Kit** | Signal la irundhu fake scan/photo images generate pannum | Oru oru factor ah control panni test panna |
| **ECG-Image-Database** | 37,191 images / 2,243 records (Germany, USA, Norway) | Real printed/photo images vechu evaluation |
| **Indian printouts** | Route A: print panni photo | Phase II |

**Tools:** Python, Google Colab, wfdb (PTB-XL read panna), OpenCV (grid, perspective), PyTorch (inference mattum, training illa), **NeuroKit2** (main tool), GitHub, Overleaf/Zotero (paper).

**Baselines:** ECG-Digitiser (challenge winner, BSD-2 licence) and Open-ECG-Digitizer (licence "Other"; use pannradhukku munnadi LICENSE file padikkanum).

<aside>
⚠️

ECG-Image-Kit oda "photo-like" images **synthetic**, real phone photo illa. Guide kitta honest ah sollunga.

</aside>

---

# 8. Ippo status — honest ah

**✅ Mudinjadhu**

- Title and problem statement approved, 7 Sep (verbal)
- 20-paper literature survey: 10 verified, **10 innum check pannanum**
- 5 research gaps, methodology, architecture, experiment plan
- Review II deck (16 slides), Notion la ellam documented, roadmap ready

**❌ Innum aagala**

- **Code innum run pannala.** Prototype illa; slide 12 la placeholders dhaan irukku.
- Team work split finalize aagala (slide 13).
- Objectives ku guide approval venum.
- Panel sonna topic change **written record** ah venum (General Guideline 3).

---

# 9. Plan (roadmap)

| Dates | Enna |
| --- | --- |
| 14–16 Sep | Approvals + oru baseline run panni scan vs photo SNR chart |
| **17 Sep** | **Review II** |
| 18–24 Sep | Dev/test split, benchmark script, error decomposition (A1) |
| 22 Sep – 1 Oct | Clinical-parameter metric + second baseline |
| 25 Sep – 4 Oct | Oru oru factor ah test panra robustness study |
| 2–5 Oct | Test set run, code release |
| **5 Oct onwards** | **Paper writing start** |
| **10 & 12 Oct** | **Review III** (publication status check) |

Full step-by-step: [Roadmap — Steps to the Paper (14 Sep → 12 Oct 2026)](Roadmap%20%E2%80%94%20Steps%20to%20the%20Paper%20(14%20Sep%20%E2%86%92%2012%20Oct%202026%203db2caadb481819e954ef911bed199c2.md)

---

# 10. Naalaikku guide kitta kekka vendiyadhu

- [ ]  **5 objectives approve pannuveengala?**
- [ ]  "Risk prediction" illa, **"diagnostic classification"** nu solrom; PTB-XL la patient follow-up data illa. Idhu OK va?
- [ ]  **Phase I = digitization + metric, classification Phase II** dhaan. Idhu OK va?
- [ ]  Panel topic change sonnadha **Log Book / Review I remarks la eludhi sign** vaangalama?
- [ ]  Work split OK va? **Rameshkumar:** pipeline · **Niranjana:** data/images · **Risvanth:** clinical metric
- [ ]  Paper ku endha type venue prefer panreenga? Conference ah (CinC maadhiri) illa journal ah (BSPC, CMPB, Physiological Measurement)? Scopus indexing namma verify pannuvom.
- [ ]  Review II **17 Sep** dhaan confirm ah? Department schedule la 14–15 Sep nu irukku.

<aside>
📝

Guide sonna answers ah inga eludhunga, date oda. Apram hub page la approval checkbox tick pannunga.

</aside>

---

# 11. Kekka vaaippu irukkura questions & answers

| Question | Answer |
| --- | --- |
| Phone photo already digitize panna mudiyaadha? | Mudiyum. **PMcardio** nu oru commercial tool irukku, aana closed source. Organisers test pannapo adhoda performance image type ku image type maaruchu. **Open, reproducible** method photo ku per-type results report pannala. |
| Yen puthu model build pannala? | 19.65 dB scan la, funded lab clinical data vechu vaangunadhu. Naama yaarum number publish pannadha idathula compete panrom: clinical-unit error, per-type photo results. |
| Yen Stage 4 target? | Grid dhaan units kudukkura ore thing. Photo la grid spacing page muzhukka maarum, so oru global scale factor thappa dhaan varum. Aana line paaka correct ah irukkum. |
| Yen risk prediction illa? | PTB-XL la diagnosis labels mattum dhaan irukku, follow-up/outcome data illa. |
| Patient data, ethics? | Route A: PTB-XL signal ah print panni photo edukrom. Patient data illa. |
| Topic yen maaruchu? | Review I panel dhaan maatha sonnanga, indha topic um avanga dhaan kuduthaanga. |
| AI use pannengala? | Aamaa, acknowledge panrom. Ella number um primary source la check pannirukom, and each member avanga part ah explain panna mudiyum. |

---

# 12. Idhu mattum sollaadheenga ❌

<aside>
🚫

- "Yaaralum phone photo digitize panna mudiyadhu" → PMcardio pannudhu.
- "State of the art ah beat pannitom" → innum edhuvum measure pannala.
- "Risk prediction" → data support pannala.
- "Clinically validated" → doctor verification plan illa.
- "First time ECG image digitize panrom" → 2005 la ECGScan irukku.
</aside>

<aside>
✅

**Idha sollunga:** *"Open, reproducible, independently evaluated method edhuvum phone photo ku per-image-type results report pannala. Indian printouts la, clinical units la yaarum evaluate pannala."*

</aside>

---

# 13. 1-minute la guide kitta solla (ready script)

> "Sir, namma project phone la edutha ECG printout photo va signal ah maathradhu. PhysioNet Challenge 2024 la winner team kooda scan la +4.9 dB vaangunaanga, aana phone photo la ella category layum negative SNR. Organisers avangaloda paper la 'SNR clinical measurement ah reflect pannala' nu solli irukanga. Namma main contribution adhu dhaan: digitization error ah doctor use panra units la, HR in BPM, PR/QRS/QT in ms, ST in mV, NeuroKit2 vechu measure pannradhu. Idhukku GPU um training um venam. Grid scale extraction (Stage 4) dhaan photo la fail aagura main point nu nenaikkirom; adha experiment panni prove panna porom. Data PTB-XL, images ECG-Image-Kit. Classification Phase II la. Ippo methodology, architecture ready; baseline run indha week la panrom. Objectives, work split, venue pathi ungaloda approval venum sir."
> 

---

<aside>
🤖

**AI acknowledgement (Student Guideline 9):** indha explanation AI assistance oda eludhunadhu. Guide kitta pesaradhukku munnadi each member padichu purinjukkanum.

</aside>