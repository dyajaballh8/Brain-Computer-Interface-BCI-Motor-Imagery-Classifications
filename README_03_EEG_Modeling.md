# README — 03_EEG_Modeling.ipynb
## EEG Modeling, Benchmarking & Model Evaluation Pipeline

هدف النوتبوك: **بناء وتدريب وتقييم نماذج تعلم الآلة (Classical ML) والتعلم العميق (Deep Learning)** لتصنيف تخيل الحركة (Motor Imagery: `Left` vs `Right`) بدقة عالية، اختيار أفضل موديل وتأكيد صحته عصبياً وتصديره للإنتاج.

---

## Cell 1 — Configuration, Imports & Data Loading

**ايه اللي اتعمل:**
استيراد المكتبات الأساسية، تثبيت بذور العشوائية (Seeds) لضمان **reproducibility** تامة، وتحميل مصفوفات الـ EEG النظيفة من مرحلة المعالجة `processed_data/`.

| المتغير | القيمة | الغرض |
|---|---|---|
| `X_eeg.npy` | `(2400, 8, 1024)` | مصفوفة الإشارات المعالجة (Trials × Channels × Samples) |
| `y_eeg.npy` | `(2400,)` | التصنيفات (0 = Left, 1 = Right) |
| `FS` | 256 Hz | تردد أخذ العينات |
| `CHANNELS` | 8 قنوات | `['C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2']` |

**الـ insight:**
- توازن الفئات مثالي: Left = 1187 (49.5%)، Right = 1213 (50.5%) ✅
- لا نحتاج إلى أساليب معالجة عدم التوازن مثل SMOTE أو Class Weighting.

---

## Cell 2 — Stratified Train / Test Split

**ايه اللي اتعمل:**
تقسيم الـ 2400 trial إلى:
- **80% Train / Validation (1920 trials):** مخصصة حصرياً للـ Cross-Validation وتدريب النماذج وضبط الـ Hyperparameters.
- **20% Hold-out Test Set (480 trials):** معزولة تماماً كـ Unseen Data لتقييم التعميم النهائي.

| المجموعة | عدد الـ Trials | Left (0) | Right (1) | الغرض |
|---|---|---|---|---|
| **Train / Val** | **1920** | 950 | 970 | 5-Fold Cross-Validation والتدريب |
| **Hold-out Test** | **480** | 237 | 243 | التقييم النهائي المستقل |

**الـ insight:**
- استخدام `stratify=y` يضمن الحفاظ على نفس النسبة المتوازنة بين الـ Train والـ Test لمنع أي تحيز في التقييم.

---

## Cell 3 — Feature Engineering Pipeline

**ايه اللي اتعمل:**
استخراج **24 خاصية** مبنية على المعايير العلمية للـ BCI وعلم وظائف الأعصاب (Neurophysiology):

1. **Sub-band Common Spatial Pattern (FBCSP):**
   - فلترة الإشارة إلى نطاق $\mu$ (8-12 Hz) ونطاق $\beta$ (16-24 Hz).
   - استخراج مصفوفات التباين المكاني (Spatial Filters) التي تعظّم التباين بين اليد اليمنى واليسرى (4 خصائص).
2. **Spectral Band Power (Welch PSD):**
   - حساب الطاقة اللوغاريتمية لنطاق $\mu$ ونطاق $\beta$ عبر جميع القنوات الثمانية (16 خاصية).
3. **Hemispheric Motor Asymmetry (فروق نصفي المخ):**
   - قياس عدم التماثل الحركي الحاسم بين القشرة الحركية اليسرى واليمنى:
     - $\Delta \text{Mu}_{C3-C4} = \ln(P_{C3}) - \ln(P_{C4})$
     - $\Delta \text{Beta}_{C3-C4} = \ln(P_{C3}) - \ln(P_{C4})$
     - بالإضافة إلى الفروق الجبهية والجدارية ($\Delta F3-F4$, $\Delta P3-P4$).

**الـ insight:**
- دمج الـ CSP مع الـ PSD وفروق الـ C3/C4 يمنح النماذج الكلاسيكية قدرة عالية على التقاط ظاهرة الـ ERD (Event-Related Desynchronization).

---

## Cell 4 — Multi-Model Benchmarking (5-Fold Stratified CV)

**ايه اللي اتعمل:**
تدريب واختبار 6 نماذج كلاسيكية تحت **5-Fold Stratified Cross-Validation** صارمة. داخل كل Fold، يتم تدريب فلاتر الـ CSP والـ Scaler حصرياً على بيانات التدريب لضمان **Zero Data Leakage**.

**نتائج الـ Cross-Validation (النماذج الكلاسيكية):**

| النموذج | النوع | CV Accuracy | CV ROC-AUC | CV F1-Macro |
|---|---|---|---|---|
| **SVM (RBF Kernel)** | Classical ML | **74.37% ± 3.37%** | **0.8152** | **0.7437** |
| **Linear SVM** | Classical ML | 74.27% ± 3.77% | 0.8122 | 0.7426 |
| **Random Forest** | Classical ML | 73.91% ± 3.00% | 0.8163 | 0.7390 |
| **LightGBM** | Classical ML | 73.91% ± 2.94% | 0.8172 | 0.7389 |
| **Logistic Regression** | Classical ML (Baseline) | 73.80% ± 3.97% | 0.8118 | 0.7379 |
| **XGBoost** | Classical ML | 73.13% ± 2.91% | 0.8155 | 0.7312 |

**الـ insight:**
- **SVM بـ RBF Kernel** حقق أفضل أداء بين النماذج الكلاسيكية، يليه الـ Linear SVM ثم الـ Random Forest.
- حتى أبسط نموذج (Logistic Regression) تجاوز 73.8%، مما يؤكد جودة الخصائص المستخرجة مقارنة بمستوى الصدفة (50%).

---

## Cell 5 — Deep Learning Benchmark (EEGConvNet / PyTorch)

**ايه اللي اتعمل:**
بناء شبكة عصبية التفافية أحادية البعد (**1D-CNN / EEGConvNet**) مخصصة لبيانات الـ EEG تعمل مباشرة على الإشارة الزمنية `(Batch, 8 Channels, 1024 Samples)`:
- **Temporal Conv:** فلاتر زمنية بطول 32 عينة لالتقاط ترددات $\mu$ و $\beta$.
- **Spatial Conv:** دمج المعلومات عبر القنوات لاستشعار تباين C3 و C4.
- **Adaptive Average Pooling & Dropout:** لمنع الـ Overfitting.

**لوحة الصدارة الكاملة (Full Leaderboard):**

| النموذج | النوع | CV Accuracy | CV ROC-AUC | CV F1-Macro |
|---|---|---|---|---|
| 🥇 **EEGConvNet (PyTorch)** | **Deep Learning** | **86.41% ± 3.04%** | **0.9635** | **0.8625** |
| 🥈 **SVM (RBF Kernel)** | Classical ML | 74.37% ± 3.37% | 0.8152 | 0.7437 |
| 🥉 **Linear SVM** | Classical ML | 74.27% ± 3.77% | 0.8122 | 0.7426 |
| **LightGBM** | Classical ML | 73.91% ± 2.94% | 0.8172 | 0.7389 |
| **Random Forest** | Classical ML | 73.91% ± 3.00% | 0.8163 | 0.7390 |
| **Logistic Regression** | Classical ML | 73.80% ± 3.97% | 0.8118 | 0.7379 |
| **XGBoost** | Classical ML | 73.13% ± 2.91% | 0.8155 | 0.7312 |

**الـ insight:**
- الـ **EEGConvNet** تفوق بفارق كبير (+12%) على كافة النماذج الكلاسيكية محققاً **86.41% CV Accuracy** و **0.9635 ROC-AUC**؛ لأن الشبكة تتعلم الفروق الطورية والزمنية الدقيقة عبر القنوات الثمانية بدون فقدان المعلومات الناتج عن حساب المتوسطات الإحصائية.

---

## Cell 6 — Hyperparameter Tuning (Top Classical Model)

**ايه اللي اتعمل:**
تحسين أفضل نموذج كلاسيكي (SVM RBF) باستخدام **GridSearchCV** عبر فضاء البارامترات:
- معيار التنظيم $C \in [0.5, 1.0, 2.0, 5.0]$
- معامل الـ Kernel $\gamma \in [\text{'scale'}, 0.02, 0.05, 0.1]$

**النتيجة:**
- **أفضل بارامترات:** `{'C': 1.0, 'gamma': 'scale'}`
- **دقة الـ CV بعد الضبط:** ارتفعت إلى **75.36%** ✅

---

## Cell 7 — Final Evaluation on Hold-Out Test Set (480 Unseen Trials)

**ايه اللي اتعمل:**
تقييم أفضل نموذج كلاسيكي (Tuned SVM) وأفضل نموذج تعلم عميق (EEGConvNet) على مجموعة الاختبار المستقلة (Hold-out Test Set) التي لم يرها أي موديل مسبقاً.

### مقارنة الأداء على الـ Test Set:

| المقياس | Tuned SVM (RBF) | EEGConvNet (PyTorch) | الفارق لصالح Deep Learning |
|---|---|---|---|
| **Test Accuracy** | 72.71% | **92.92%** | **+20.21%** 🚀 |
| **Balanced Accuracy** | 72.75% | **92.94%** | **+20.19%** |
| **F1-Score (Macro)** | 0.7269 | **0.9292** | **+0.2023** |
| **ROC-AUC** | 0.7975 | **0.9794** | **+0.1819** |
| **Left Precision / Recall** | 0.71 / 0.76 | **0.91 / 0.95** | تمييز شبه مثالي |
| **Right Precision / Recall**| 0.75 / 0.70 | **0.94 / 0.91** | تمييز شبه مثالي |

**الـ insight:**
- نموذج **EEGConvNet** حقق أداءً استثنائياً بنسبة دقة **92.92%** على بيانات غير مرئية تماماً، مع مساحة تحت منحنى الـ ROC بلغت **0.9794**.
- مصفوفات الالتباس (Confusion Matrices) تظهر توازناً ممتازاً في التنبؤ بكلتا اليدين دون أي انحياز لفئة على حساب الأخرى.

---

## Cell 8 — Neurophysiological Validation & Model Export

**ايه اللي اتعمل:**
1. **التحقق العصبي والفيزيولوجي (Neurophysiological Validation):**
   - تحليل أهمية الخصائص (Feature Importance) أكد أن أهم الخصائص الحاكمة في القرار هي:
     - `Asym_Mu_C3_C4` (فرق طاقة نطاق الميو بين C3 و C4) في المركز الأول.
     - خصائص `CSP_Mu_1` و `CSP_Beta_1`.
     - طاقة `PSD_Mu_C3` و `PSD_Mu_C4`.
   - هذا يثبت 100% أن النموذج يتخذ قراراته بناءً على القواعد العصبية الحقيقية لـ Motor Imagery (تثبيط C4 عند تخيل اليد اليسرى، وتثبيط C3 عند تخيل اليد اليمنى) وليس على ضوضاء عشوائية.

2. **تصدير الموديلات (Artifacts Export):**
   - تم حفظ الـ Pipelines الكاملة في مجلد `models/`:
     - `models/best_classical_pipeline.joblib`: الموديل الكلاسيكي المكتمل مع الـ CSP والـ Scaler.
     - `models/best_deep_model.pt`: أوزان شبكة الـ PyTorch المدربة.
     - `models/leaderboard.csv`: جدول مقارنة النماذج.
     - `models/test_metrics.json`: ملخص مقاييس الاختبار بصيغة JSON.
     - الرسوم البيانية: `confusion_matrices.png` و `roc_curves.png` و `feature_importance.png`.

---

## ملخص مرحلة النمذجة

```
Input:               2400 clean trials (8 channels x 1024 samples)
Train/Test Split:    1920 Train (80%) vs 480 Test (20%) - Stratified
Validation:          5-Fold Stratified Cross-Validation (Zero Data Leakage)

Best Classical:      Tuned SVM (RBF)  → Test Acc: 72.71% | ROC-AUC: 0.7975
Best Deep Learning:  EEGConvNet       → Test Acc: 92.92% | ROC-AUC: 0.9794

🏆 Winner:           EEGConvNet (PyTorch) كأعلى دقة، مع الاحتفاظ بـ Tuned SVM كنموذج فائق التفسير.
```

> [!TIP]
> للاستخدام العملي، استخدم `best_deep_model.pt` عندما تكون الأولوية للدقة القصوى في التطبيقات اللحظية (Real-time BCI)، واستخدم `best_classical_pipeline.joblib` عندما تكون هناك حاجة لتفسير بيولوجي مباشر لكل مكوّن.
