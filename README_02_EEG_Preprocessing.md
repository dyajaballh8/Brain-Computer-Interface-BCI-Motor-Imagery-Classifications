# README — 02_EEG_Preprocessing.ipynb
## EEG Preprocessing Pipeline

هدف النوتبوك: **تحويل ملفات الـ Excel الخام إلى arrays نظيفة موحدة** جاهزة للتدريب — فلترة، تطبيع، رفض الـ trials الوحشة، وحفظ الناتج.

---

## Cell 1 — Parameters

**ايه اللي اتعمل:**
استيراد المكتبات وتعريف كل بارامترات الـ preprocessing في مكان واحد.

| المتغير | القيمة | الغرض |
|---|---|---|
| `BP_LOW / BP_HIGH` | 8 / 30 Hz | نطاق الـ band-pass (mu + beta) |
| `TARGET_DURATION_SEC` | 4.0 ثانية | الطول الموحد لكل trial |
| `MIN_DURATION_SEC` | 2.0 ثانية | الحد الأدنى لقبول الـ trial |
| `MAX_NAN_FRAC` | 5% | رفض الـ trial لو أكتر من 5% NaN |
| `FLAT_STD` | 1e-6 | رفض الـ trial لو فيه قناة flat |
| `PTP_MAD_K` | 5.0 | مضاعف MAD للـ amplitude rejection |
| `ZSCORE_MODE` | `channel` | Z-score لكل قناة بشكل منفصل |
| `TARGET_LEN` | **1024 samples** | 4.0 × 256 Hz |

**الـ insight:** استخدام `channel` mode في الـ Z-score يحافظ على العلاقة النسبية بين القنوات ويزيل اختلاف الـ baseline لكل قناة على حدة.

---

## Cell 2 — Channel Alignment

**ايه اللي اتعمل:**
قراءة header كل الملفات الـ 2400 بالتوازي، واختيار الـ layout الأكثر شيوعاً كـ reference، وإعادة ترتيب كل trial وفقاً له.

**الـ insight:**
- **Reference channels (8):** `['C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2']`
- **Files بـ layout مختلف: 0** ✅ — كل الـ trials لها نفس القنوات بنفس الترتيب
- C3 و C4 موجودتان ✅ (شرط أساسي للتحليل الحركي)
- هذه الخطوة تحمي من أي trial جاي بقنوات مختلفة أو ناقصة

---

## Cell 3 — Cleaning & Filtering Functions

**ايه اللي اتعمل:**
تعريف دوال معالجة الإشارة اللي بتُطبَّق على كل trial.

**الدوال:**

| الدالة | الوظيفة |
|---|---|
| `interpolate_nans` | ملء فجوات الـ NaN بالاستيفاء الخطي (Linear interpolation) |
| `common_average_reference` | طرح متوسط كل القنوات من كل قناة (يقلل الـ noise المشترك) |
| `notch_filter` | فلتر IIR يزيل ضوضاء الكهرباء عند 50 Hz |
| `bandpass_filter` | Butterworth band-pass من 8 إلى 30 Hz (نطاق mu/beta) |

**الـ insight:**
- ترتيب التطبيق: `CAR → Notch → Bandpass` — ترتيب صحيح ومعياري في BCI
- استخدام `sosfiltfilt` (zero-phase) يمنع أي تأخر زمني في الإشارة ✅
- الـ CAR يُحسِّن نسبة الإشارة للضوضاء خاصةً لقنوات C3/C4

---

## Cell 4 — Length & Scale Standardization

**ايه اللي اتعمل:**
تعريف دالة `standardize_trial` التي توحّد طول وحجم كل trial.

**الخطوات بالترتيب:**
1. **Crop** — قصّ الـ trial على 1024 sample لو أطول
2. **Z-score** — تطبيع لكل قناة: `(x - mean) / (std + 1e-8)` على البيانات الحقيقية بس
3. **Zero-pad** — لو الـ trial أقصر من 1024، يُكمَّل بأصفار في النهاية (بعد التطبيع)

**الـ insight:**
- الـ padding بيتم **بعد** الـ Z-score عشان الأصفار الإضافية ماتأثرش على الإحصاءات ✅
- الـ `1e-8` يمنع القسمة على صفر لو القناة ثابتة تماماً
- الناتج لكل trial: `(1024 samples × 8 channels)` ثم يُحوَّل لـ `(8 × 1024)` للتوافق مع PyTorch/ML

---

## Cell 5 — Batch Processing & Artifact Rejection

**ايه اللي اتعمل:**
تشغيل الـ pipeline الكامل على كل الـ 2400 ملف بالتوازي، ثم رفض الـ trials ذات الـ amplitude المتطرف.

**خطوات `process_trial`:**
```
read file → check channels → check length → check NaNs
→ interpolate NaNs → check flat channels
→ CAR → notch → bandpass → crop/pad → Z-score
```

**ثم (على مستوى الـ dataset):**
- حساب Peak-to-Peak amplitude لكل trial
- رفض كل trial: `ptp > median + 5 × MAD`

**الـ insight:**

| المرحلة | Trials المرفوضة | السبب |
|---|---|---|
| too_short | 0 | كل الملفات 4 ثواني ✅ |
| too_many_nans | 0 | لا يوجد NaN ✅ |
| flat_channel | 0 | كل القنوات نشطة ✅ |
| artifact_amplitude | 0 | لا توجد amplitudes متطرفة ✅ |
| **kept** | **2400** | **100% من الـ trials** ✅ |

- `X.shape = (2400, 8, 1024)` — `(trials, channels, samples)` ✅
- `y.shape = (2400,)` — 0 = Left, 1 = Right ✅
- **كل الـ 2400 trial اجتازت الـ QC** ✅

---

## Cell 6 — Before vs After Plot

**ايه اللي اتعمل:**
مقارنة بصرية على قناة C3: trial واحدة Left وواحدة Right — الإشارة الخام (عمود يسار) مقابل الإشارة المُعالَجة (عمود يمين).

**الـ insight:**
- الإشارة الخام: نطاق واسع، noise عالي، قيم كبيرة (µV)
- الإشارة المُعالَجة: نظيفة، مُركَّزة في 8-30 Hz، مُطبَّعة حول صفر (Z-scored)
- يتضح بصرياً الفرق في طاقة الإشارة بين Left و Right على C3 — نتيجة الـ ERD ✅
- التحقق يؤكد إن الـ pipeline اشتغل صح

---

## Cell 7 — Save Outputs

**ايه اللي اتعمل:**
حفظ كل نواتج الـ preprocessing في مجلد `processed_data/`.

**الملفات المحفوظة:**

| الملف | الوصف |
|---|---|
| `X_eeg.npy` | Array الـ EEG نظيفة — shape: `(2400, 8, 1024)` |
| `y_eeg.npy` | Labels — shape: `(2400,)` — 0=Left, 1=Right |
| `channels.json` | أسماء الـ 8 قنوات بالترتيب |
| `config.json` | كل الـ parameters المستخدمة (FS, bands, etc.) |
| `kept_trials.csv` | قائمة الـ trials المقبولة مع labels |
| `excluded_trials.csv` | قائمة الـ trials المرفوضة مع سبب الرفض |

**الـ insight:**
- Reload check ناجح ✅ — الملفات اتحفظت وقُرئت صح
- `config.json` يضمن **reproducibility** — تقدر دايماً تعرف بالضبط إيه اللي اتعمل
- `kept_trials.csv` و `excluded_trials.csv` يسمحان بـ **auditability** كاملة

---

## ملخص النوتبوك

```
Input:  2400 Excel files (1024 samples × 8 channels each)
Output: X_eeg.npy → shape (2400, 8, 1024) | dtype float32
        y_eeg.npy → shape (2400,)          | dtype int64

Pipeline:  CAR → 50Hz Notch → 8-30Hz Bandpass → Crop(4s) → Z-score
Rejected:  0 trials (100% kept)
Classes:   Left=1187, Right=1213 (balanced ✅)

→ الداتا جاهزة للـ Model Training
```

> [!TIP]
> لو عندك داتا حقيقية، الـ pipeline هيشتغل عليها تلقائياً بنفس الـ parameters — بس تأكد إن FS صح وإن القنوات C3/C4 موجودة.
