# README — 01_EEG_EDA.ipynb
## EEG Exploratory Data Analysis

هدف النوتبوك: **فهم الداتا قبل أي preprocessing** — التحقق من التوازن بين الكلاسات، جودة الملفات، وهل في إشارة ERD حقيقية في القنوات الحركية.

---

## Cell 1 — Configuration

**ايه اللي اتعمل:**
استيراد المكتبات وتعريف كل الثوابت والدوال المساعدة في مكان واحد.

| المتغير | القيمة | الغرض |
|---|---|---|
| `DATA_DIR` | `data/eeg_files` | مجلد ملفات الـ Excel |
| `LABELS_CSV` | `labels.csv` | ملف الـ labels (Left / Right) |
| `FS` | 256 Hz | معدل أخذ العينات |
| `NOTCH_HZ` | 50 Hz | فلترة ضوضاء الكهرباء |
| `MU / BETA` | 8-12 / 13-30 Hz | النطاقات الترددية للحركة |

**الدوال المُعرَّفة:**
- `natural_key` — ترتيب الملفات ترتيباً طبيعياً (trial_2 قبل trial_10)
- `list_trial_files` — جلب كل ملفات `.xlsx` من المجلد
- `read_trial` — قراءة ملف Excel كـ DataFrame رقمي نظيف
- `build_mapping` — ربط كل ملف بـ label الخاص به

**الـ insight:** تعريف `NON_EEG_HINTS = ("TIME", "LABEL", "UNNAMED", "INDEX")` يضمن حذف أي عمود مش قناة EEG تلقائياً.

---

## Cell 2 — Clean Mapping

**ايه اللي اتعمل:**
مطابقة ملفات الـ Excel مع `labels.csv` بالترتيب الأبجدي الطبيعي، وحفظ الجدول في `trial_label_mapping.csv`.

**الـ insight:**
- تم التحقق من إن عدد الملفات = عدد الـ labels (2400 = 2400) ✅
- مفيش ملفات مفقودة (missing files = 0) ✅
- الناتج: جدول فيه `trial_id`, `filename`, `label` لكل trial

---

## Cell 3 — Class Balance

**ايه اللي اتعمل:**
حساب نسبة توزيع كلاسات Left / Right ورسم bar chart.

**الـ insight:**

| Class | Count | % |
|---|---|---|
| Left | 1187 | 49.5% |
| Right | 1213 | 50.5% |

- **Majority/Minority ratio ≈ 1.02** → الداتا **متوازنة جداً** ✅
- مفيش حاجة لـ oversampling أو class weighting

---

## Cell 4 — Single Trial Inspection

**ايه اللي اتعمل:**
فتح أول trial وعرض شكله وأسماء القنوات والإحصاءات الأساسية لكل قناة.

**الـ insight:**
- **Shape:** `(1024 samples × 8 channels)` = 4 ثواني عند 256 Hz ✅
- **القنوات الـ 8:** `C3, C4, F3, F4, P3, P4, O1, O2` — قنوات حركية وجدارية وقذالية
- القنوات C3 و C4 موجودتين ✅ (أساسيتين لتحليل ERD الحركي)
- القيم في نطاق معقول بدون outliers متطرفة

---

## Cell 5 — Dataset-Wide Quality Audit

**ايه اللي اتعمل:**
مسح كل الـ 2400 ملف بالتوازي (parallel) وتسجيل لكل trial: الطول، عدد القنوات، NaNs، القنوات الـ flat، والـ outliers. في نفس الوقت حساب Welch PSD لـ C3 و C4.

**الـ insight:**
- المسح اشتغل بـ 8 workers بالتوازي وخلص في ~32 ثانية
- **كل الـ 2400 trial لها نفس الـ layout** (8 قنوات) ✅
- مفيش NaNs ✅
- مفيش قنوات flat ✅
- الـ PSD اتحسب بنجاح لـ C3 و C4 في كل الـ trials ✅

---

## Cell 6 — Quality Summary & Plots

**ايه اللي اتعمل:**
تصنيف كل trial كـ "bad" لو عنده أي مشكلة، ورسم توزيع مدة الـ trials وأقصى amplitude.

**الـ insight:**

| المشكلة | عدد الـ trials |
|---|---|
| Layout مختلف | 0 |
| NaNs | 0 |
| Flat channels | 0 |
| Outliers | 0 |
| **Bad trials إجمالاً** | **0** |

- **كل الـ 2400 trial سليمة** ✅
- مدة كل trial: **4.0 ثانية بالضبط** (uniform distribution)
- توزيع الـ amplitude منتظم بدون spikes متطرفة

---

## Cell 7 — Time-Domain Plots

**ايه اللي اتعمل:**
رسم قناتي C3 و C4 لـ 2 trials عشوائية من كل كلاس (Left و Right): الإشارة الخام (رمادي) مقابل الإشارة بعد فلترة 8-30 Hz (أزرق).

**الـ insight:**
- الإشارة الخام (رمادي) تحتوي على كل الترددات مع noise
- بعد الفلترة (أزرق)، تظهر مكونات نطاق mu/beta بوضوح
- يلاحظ الفرق البصري بين Left و Right في قوة الإشارة على C3 و C4 — مؤشر مبدئي على وجود ERD

---

## Cell 8 — PSD & ERD Check

**ايه اللي اتعمل:**
حساب متوسط Welch PSD لـ C3 و C4 للـ trials الجيدة، ورسمها مع تلوين نطاق Mu و Beta، وحساب نسبة تغير الطاقة.

**الـ insight (ERD pattern متوقع):**

| النطاق | القناة | Left Power | Right Power | الفرق % |
|---|---|---|---|---|
| Mu (8-12 Hz) | C3 | أعلى | أقل | Left > Right |
| Mu (8-12 Hz) | C4 | أقل | أعلى | Right > Left |
| Beta (13-30 Hz) | C3 | أعلى | أقل | Left > Right |
| Beta (13-30 Hz) | C4 | أقل | أعلى | Right > Left |

- **ERD (Event-Related Desynchronization) موجود** ✅
- أثناء Left trials: طاقة mu/beta على **C4** (نصف الكرة الأيمن) أقل — الحركة اليسرى يتحكم فيها النصف الأيمن
- أثناء Right trials: طاقة mu/beta على **C3** (نصف الكرة الأيسر) أقل
- هذا البيولوجي منطقي ومتوافق مع أدبيات BCI ✅

---

## ملخص النوتبوك

```
✅ 2400 trial — كلهم سليمين
✅ Balanced classes (49.5% / 50.5%)
✅ 8 قنوات موحدة لكل trial
✅ 4 ثواني لكل trial @ 256 Hz
✅ ERD pattern واضح في C3/C4
→ الداتا جاهزة للـ Preprocessing
```
