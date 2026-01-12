
# Список используемой литературы:

1. Рашка С., Лю Ю., Мирджалили В. Машинное обучение с PYTORCH и SCIKIT-LEARN. - Астана: Foliant, 2024. - 688 c. 

Книга содержит руководство по машинному и глубокому обучению с использованием языка программирования Python, фрейморка Pythorch и библиотеки Scikit-learn, рассмотрены основы машинного обучения, алгоритмы для задач классификации, предварительная обработка и сжатие данных, современные методы оценки моделей и объединение различных моделей для ансамблевого обучения.

2. Машинное обучение с использованием Python. Сборник рецептов. Практические решения от предобработки до глубокого обучения. Галлатин К., Элбон К.,  - Астана: Алист, 2024. - 448 с. 

Книга содержит задачи машинного обучения, такие как загрузка и обработка текстовых и числовых данных, отбор моделей и приведены практические решения.

## Научные статьи:

1. Balikci Ci̇cek I, Kucukakcali Z. Analysis of voice data based onassociative classification rules for early diagnosis of Parkinson'sdisease. Med Science. 2025;14(2):355-61.https://www.researchgate.net/publication/392411659_Analysis_of_voice_data_based_on_associative_classification_rules_for_early_diagnosis_of_Parkinson's_disease
Статья о применении методов машинного обучения для ранней диагностики болезни Паркинсона.

**Датасет**: https://archive.ics.uci.edu/dataset/174/parkinsons (как у нас сейчас, отсюда же можем взять описания некоторых признаков), 195 голосовых записей от 31 человека, 23 из которых имеют диагноз БП.

**Модель**: в качестве модели классификации используется ассоциативная классификация (Associative Classification). Работа в два этапа: дискретизация непрерывных признаков по методу Ameva; генерация решающих правил по методам Apriori или FP-Growth.

**Результат**: Balanced Accuracy – 0.952; recall – 0.905; precision – 1; F1-score – 0.95.

2.https://www.researchgate.net/publication/370894866_Classification_of_Parkinson's_Disease_from_Voice_-_Analysis_of_Data_Selection_Bias
Статья о применении методов машинного обучения для ранней диагностики болезни Паркинсона.

**Датасет**: https://www.synapse.org/mPower (из России не открывается). Аудиозаписи произношения фонемы “Аа”, собранные от ~25000 пациентов (соотношение примерно 10к БП / 15к нет БП) с различными способами выделения признаков из аудиозаписей.

**Модель**: в качестве моделей классификации используются SVM (ядро не указано, скорее всего ‘rbf’) и CatBoost. Акцент сделан на сбалансированном учете пациентов по полу и возрасту на обучении и на тесте.

**Результат**: для SVM: Accuracy – 0.83; ROC-AUC – 0.82; F1-score – 0.78. 
для CatBoost: Accuracy – 0.84; ROC-AUC – 0.83; F1-score – 0.80.

3. Xu, H., Xie, W., Pang, M., Li, Y., Jin, L., Huang, F., & Shao, X. (2025). Non-invasive detection of Parkinson’s disease based on speech analysis and interpretable machine learning. Frontiers in Aging Neuroscience, 17, 1586273. https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2025.1586273

**Датасет**: https://archive.ics.uci.edu/dataset/174/parkinsons 

**Модель**: Random Forest и Gradient Boosting Machine (GBM). 

**Этапы**: обработка данных, отбор признаков (LASSO), балансировка классов (SMOTE).
Ключевые акустические признаки: jitter, shimmer, NHR, RPDE и PPE

**Результат**: AUC-ROC 0.98, recall 0.95

4. Wiharto, W., Sucipto, A., & Salamah, U. (2023). Hilbert-Schmidt Independence Criterion Lasso Feature Selection in Parkinson’s Disease Detection System. International Journal of Fuzzy Logic and Intelligent Systems, 482–499. https://www.researchgate.net/publication/397095201_Hilbert-Schmidt_Independence_Criterion_Lasso_Feature_Selection_in_Parkinson's_Disease_Detection_System

**Датасет**: https://archive.ics.uci.edu/dataset/174/parkinsons
https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification
252чел (188 с БП)→ 756 записей итого. Использовались акустические параметры типа jitter, shimmer, частотные характеристики на устойчивом звуке "ааа".

**Модель и метод**:  Support Vector Machine (SVM), HSIC Lasso.

**Результат**: первый датасет- Accuracy = 88.34%, второй датасет- Accuracy = 96.16%.

5. Karabayir, I., Goldman, S. M., Pappu, S., & Akbilgic, O. (2020). Gradient boosting for Parkinson’s disease diagnosis from voice recordings. BMC Medical Informatics and Decision Making, 20(1), 228. https://doi.org/10.1186/s12911-020-01250-7

**Датасет**: https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification
252чел (188 с БП)→ 756 записей итого. Использовались акустические параметры типа jitter, shimmer, частотные характеристики на устойчивом звуке "ааа".

**Модель и методы**: Сравнение градиентного бустинга (CatBoost) с Random Forest, SVM, логистической регрессией. Основной фокус – на ансамблевых методах.

**Этапы** *(на примере логистической регрессии)*: масштабирование признаков по Z-score, стратифицированное разделение, регулировка параметра C, кросс-валидация,

**Результат**: Градиентный бустинг (CatBoost) AUC-ROC = 0.938), логистическая регрессия: 0.889

6. Voiceprint recognition of Parkinson Patients Based on Deep Learning / Z. Xu–2018 https://www.researchgate.net/publication/329734178_Voiceprint_recognition_of_Parkinson_patients_based_on_deep_learning

7. Naeem, I. Voice biomarkers as prognostic indicators for Parkinson's disease using machine learning techniques / I. Naeem, A. Ditta, T. Mazhar, M. Anwar, M. M. Saeed, H. Hamam // Scientific Reports. — 2025. — Vol. 15. — Art. 12229. https://doi.org/10.1038/s41598-025-96950-3.

**Датасет**: https://archive.ics.uci.edu/dataset/174/parkinsons

**Модели**:
-Support Vector Machine (SVM)
-Random Forest (RF)
-Logistic Regression (LR)
-Decision Tree (DT)

**Методы**:
-SMOTE (Synthetic Minority Over-sampling Technique) – для борьбы с дисбалансом классов 
-PCA (Principal Component Analysis) – для снижения размерности и отбора признаков.
-Стандартизация данных (Standard Scaler).
-Удаление выбросов (методом межквартильного размаха).

Проведены 3 сценария для каждого метода:
1)обучение без smote и pca
2) smote- балансировка классов перед обучением
3) pca отбор 12 ключевых признаков 

**Результат**: Лучшую точность показала модель RF: 94.87% (сценарий со smote).

8. Machine Learning-Based Classification of Parkinson’s Disease Patients Using Speech Biomarkers. Hossain MA, Amenta F. J Parkinsons Dis.2024;14(1):95-109, PMID 38160364
Machine Learning-Based Classification of Parkinson's Disease Patients Using Speech Biomarkers - PubMed

• Описание Классификация болезни Паркинсона по акустическим признакам речи.

• **Датасет**  UCI Parkinson’s Telemonitoring / UCI Parkinson’s Speech Dataset (в статье — подмножества голосовых биомаркеров).

• **Модели**  SVM, Random Forest, Logistic Regression, kNN.

• **Методы** MFCC, jitter/shimmer, гармоничность, feature selection.

• **Результаты**  Точность около 90% (зависит от модели; SVM лучшая).

9. Voice-Based Detection of Parkinson’s Disease Using Machine and Deep Learning Approaches A Systematic Review. Sedigh Malekroodi H, Lee BI, Yi M. Bioengeeneering (Basel). 2025 Nov 20;12(11):1279, PMID 41301235
Voice-Based Detection of Parkinson's Disease Using Machine and Deep Learning Approaches: A Systematic Review - PubMed

• Описание  Обзор статей, использующих голосовые биомаркеры.

• **Датасет** UCI Parkinson’s, PC-GITA, Italian PD Voice, Czech PD Voice, Turkish PD datasets.

• **Модели** SVM, RF, CNN, LSTM, Transfer Learning‑модели.

• **Методы**  Спектрограмма, MFCC, glottal features, prosodic features.

• **Результаты**  Лучшие модели DL достигают 90–97%.

10. Transformer-based transfer learning on self-reported voice recordings for Parkinson’s disease diagnosis. Tougoi I, Zakroum M, Karrakchou O, Ghogho M. Sci Rep.2024 Dec 3;14(1):30131, PMID 39627487
Transformer-based transfer learning on self-reported voice recordings for Parkinson's disease diagnosis - PubMed

• **Датасет** Собранный crowdsourced голосовой датасет (онлайн‑записи).

• **Модели**  Wav2Vec2.0, HuBERT, XLS-R + классификатор.

• **Методы**  Transfer Learning, fine‑tuning, self-supervised embeddings.

• **Результаты** Точность до 92%.

10. Machine Learning Smart System for Parkinson Disease Classification Using the Voice as a Biomarker. Tougui I, Jiibab A, Mhamdi JE. Healthc Inform Res.2022 Jul;28(3):210-221. PMID 35982595
Machine Learning Smart System for Parkinson Disease Classification Using the Voice as a Biomarker - PubMed

• **Датасет**  UCI Parkinson’s Speech Dataset.

• **Модели**  SVM, kNN, Decision Tree.

• **Методы**  Jitter, shimmer, HNR, MFCC.

• **Результаты**  Лучший алгоритм ≈ 96%.

11. SS‑DRPL Self-supervised deep representation pattern learning for voice‑based Parkinson’s disease detection. Kim TH, Krichen M, Ojo S, Sampedro GA, Alarmo MA. Front Comput Neurosci.2024 Jun 12;18:1414462. PMID 38933392
SS-DRPL: self-supervised deep representation pattern learning for voice-based Parkinson's disease detection - PubMed

• **Датасет** PC-GITA.

• **Модели**  Self-Supervised Network + CNN classifier.

• **Методы**  Contrastive learning, deep feature extraction.

• **Результаты**  Точность 93–95%.

12. Explainable machine learning for early detection of Parkinson’s disease using vocal biomarkers. Egbo B, Nigmetolla Z, Khan NA, Jamwal PK. Front Aging Neurosci. 2025 Sep 5;17:1672971. PMID 40979153
Explainable machine learning for early detection of Parkinson's disease in aging populations using vocal biomarkers - PubMed

• **Датасет**  UCI Parkinson’s Voice Dataset.

• **Модели**  XGBoost, SVM.

• **Методы**  SHAP‑интерпретации, feature engineering.

• **Результаты**  Точность 90–94%.

13. A machine learning method to process voice samples for identification of Parkinson’s disease. Iyer A, Kemp A, Rahmatallah Y, Pillai L, Glover A, Prior F, Larson-Prior L, Virmani T. Sci Rep.2023 Nov 23; 13(1):20615. PMID 37996478

A machine learning method to process voice samples for identification of Parkinson's disease - PubMed

• **Датасет** Несколько публичных голосовых наборов данных (UCI, PC-GITA).
 
 **Модели**  CNN, SVM.

• **Методы**  Спектрограммы, голосовая сегментация.

• **Результаты**  Точность 95%+.

14. Machine Learning‑Assisted Speech Analysis for Early Detection of Parkinson’s Disease: A Study on Speaker Diarization and Classification Techniques. Di Cesare MG, Perpetuini D, Cardone D, Merla A. Sensors (Basel). 2024 Feb 26;24(5):1499. PMID 38475034
Machine Learning-Assisted Speech Analysis for Early Detection of Parkinson's Disease: A Study on Speaker Diarization and Classification Techniques - PubMed

• **Датасет**  Собственный клинический — записи речи + диаризация.

• **Модели** SVM, Decision Tree, Neural Networks.

• **Методы**  Speaker diarization, MFCC, prosodic features.

• **Результаты** Около 88–92%.

15. Harnessing Voice Analysis and Machine Learning for Early Diagnosis of Parkinson’s Disease A Comparative Study Across Three Datasets. Neto OP. J Voice. 2024 May 12:S0892-1997(24)00139-5. PMID 38740529
Harnessing Voice Analysis and Machine Learning for Early Diagnosis of Parkinson's Disease: A Comparative Study Across Three Datasets - PubMed

• **Датасет**  UCI, PC-GITA, Italian Parkinson’s Voice.

• **Модели**  SVM, Gradient Boosting, RF.

• **Методы** Вокализация гласной /a/, спектральные признаки.

• **Результаты**  Сравнение: лучший результат 95–97%.

16. Advanced comparative analysis of machine learning algorithms for early Parkinson’s disease detection using vocal biomarkers. Kumar A, Singh JP, Paygude P, Daimary R, Prasad S. Digit Health.2025 Jun 6; 11:20552076251342878. PMID 40487885
Advanced comparative analysis of machine learning algorithms for early Parkinson's disease detection using vocal biomarkers - PubMed

• **Датасет**  UCI Parkinson’s Voice Dataset.

• **Модели**  SVM, kNN, NB, ANN.

• **Методы**  Feature selection, MFCC, jitter/shimmer.

• **Результаты**  SVM ≈ 96%.
 
17. Parkinson’s Disease Detection Using Hybrid Siamese Neural Network and Support Vector Machine in Multilingual Voice Signal. Pandey PVK, Sahu SS. J Voice.2025 Aug 4:S0892-1997(25)00261-9. PMID 40764154
Parkinson's Disease Detection Using Hybrid Siamese Neural Network and Support Vector Machine in Multilingual Voice Signal - PubMed

• **Датасет**  Многоязычный собственный голосовой набор.

• **Модели**  Siamese Neural Network + SVM.

• **Методы** Speaker‑invariant embeddings, contrastive loss.

• **Результаты**  Точность ~94%.

18. Artificial Intelligence‑Based Voice Assessment of Patients with Parkinson’s Disease Off and On Treatment. Costantini G, Gesarini V, Di Leo P, Amato F, Suppa A, Asci F, Pisani A, Saggio G. Sensors (Basel). 2023 Feb 18;23(4):2293. PMID 36850893
Artificial Intelligence-Based Voice Assessment of Patients with Parkinson's Disease Off and On Treatment: Machine vs. Deep-Learning Comparison - PubMed

• **Датасет** Собственные записи пациентов до/после терапии.

• **Модели**  SVM, Deep Learning CNN.

• **Методы**  Акустические признаки, динамика голоса при лечении.

• **Результаты**  Классификация “off/on treatment” до 90%.

19. Parkinson’s Disease Detection from Voice Recordings Using Associative Memories. Luna-Ortiz I, Aldape-Perez M, Uriarte-Arcia AV, Rodriguez-Molina A, Alarcon-Peredes A, Ventura-Molina E. Healthcare (Basel). 2023 May 30; 11(11);1601. PMID 37297740
Parkinson's Disease Detection from Voice Recordings Using Associative Memories - PubMed

• **Датасет**  UCI Parkinson’s Voice Dataset.

• **Модели**  ISNDAM (Associative Memory), SVM baseline.

• **Методы**  Акустические признаки, диктор-независимая нормализация.

• **Результаты**  Точность ≈ 95%.

20. Enhancing Parkinson’s Disease Detection by Combining SMOTE and Feature Selection Using Voice Recordings. Quadir JA. J Voice.2025 Dec 16:S0892-1997(25)00478-3. PMID 41407640
Enhancing Parkinson's Disease Detection by Combining SMOTE and Feature Selection for Improved Machine Learning Classification Using Voice Recordings - PubMed

• **Датасет** UCI Parkinson’s Dataset.

• **Модели**  SVM, RF, Decision Tree.

• **Методы**  SMOTE, feature ranking.

• **Результаты** Точность до 95–96%.

21.  Explainable artificial intelligence to diagnose early Parkinson’s disease via voice analysis. Shen M, Mortezaagha P, Rahgozar A. Sci Rep.2025 Apr 5; 15(1):11687. PMID 40188263
Explainable artificial intelligence to diagnose early Parkinson's disease via voice analysis - PubMed

• **Датасет** PC-GITA.

• **Модели**  XGBoost, Ensemble ML.

• **Методы**  Explainable AI, SHAP.

• **Результаты**  Точность ~93%.

22. Explainable Ensemble and Deep Learning Framework for Parkinson’s Disease Detection from Voice Biomarkers. Aladhadh S. Diagnostics (Basel). 2025 Nov 14; 15(22):2892. PMID 41300916
An Explainable Ensemble and Deep Learning Framework for Accurate and Interpretable Parkinson's Disease Detection from Voice Biomarkers - PubMed

• **Датасет** UCI + собственные данные.

• **Модели**  CNN, Ensemble (RF + XGBoost).

• **Методы**  Spectrogram CNN, SHAP‑интерпретация.

• **Результаты** DL точность ≈ 97%.

23. Diagnosis of Parkinson’s disease based on voice using SHAP and ensemble method. Ghaheri P, Nasiri H, Shateri A, Homafar A. Comput Methods Biomech Biomed Engin. 2024 Oct; 27(13): 1858-1874. PMID 37771234
Diagnosis of Parkinson's disease based on voice signals using SHAP and hard voting ensemble method - PubMed

• **Датасет** UCI Parkinson’s “Replicated Acoustic Features”.

• **Модели** Voting Ensemble (RF + SVM + XGBoost).

• **Методы** SHAP, feature selection.

• **Результаты** Точность 98%.

24. Smartphone‑derived multimodal features including voice, finger-tapping movement and gait aid early identification of Parkinson’s disease. Lim WS, Fan SP, Chiu Si, Wu MC, Wang PH, Lin KP, Chen YM, Peng PL, Jang JR, Lin CH. NPJ Parkinsons Dis. 2025 May 5,11(1):111. PMID 40325040
Smartphone-derived multidomain features including voice, finger-tapping movement and gait aid early identification of Parkinson's disease - PubMed

• **Датасет** Смартфонные записи речи.

• **Модели**  XGBoost, RF, SVM.

• **Методы** Голос + движение + моторика → multimodal fusion.

• **Результаты** Мультимодальность улучшает точность до ~94%.

25. Voice‑Based Early Diagnosis of Parkinson’s Disease Using Spectrogram Features and AI Models. Quamar D, Ambeth Kumar VD, Rizwan M, Bagdasar O, Kadar M. Bioengeenering (Basel). 2025 Sep 29; 12(10):1052. PMID 41150550
Voice-Based Early Diagnosis of Parkinson's Disease Using Spectrogram Features and AI Models - PubMed

• **Датасет** Собственный голосовой датасет.

• **Модели** CNN, SVM.

• **Методы** Спектрограмма, MFCC.

• **Результаты** Точность до 95%.
