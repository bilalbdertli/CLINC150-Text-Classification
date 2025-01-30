# CLINC150-Text-Classification
 Text Classification with CLINC150 Dataset, using various models including  a finetuned RoBERTa-base 
 
## Project Description  
This repository contains the implementation and analysis of intent detection on the **CLINC150 dataset**, which includes 150 real-world intents. The project compares classical machine learning models (e.g., Naive Bayes, SVM), embedding-based approaches (FastText), and transformer-based models (RoBERTa). Key highlights:  
- Compared classical ML (Naive Bayes, SVM), FastText, and transformer models
- Achieved **96.22% accuracy** with fine-tuned RoBERTa-base.  
- Explored synthetic data augmentation using **In-Context Data Augmentation (ICDA)** and **Pointwise V-Information (PVI)** filtering.  
- Comprehensive evaluation of model performance, including macro F1, precision, recall, and confusion matrices.  

## Dataset  
**CLINC150** is a balanced dataset with 15,000 training, 3,000 validation, and 4,500 test examples. It covers diverse domains such as banking, travel, and utilities.  


## Results  

### Model Performance Comparison  
| Model              | Test Accuracy | Macro F1 | Training Time     |  
|--------------------|---------------|----------|-------------------|  
| Naive Bayes        | 81.62%        | 81.27%   | ~10 sec (CPU)     |  
| FastText           | 90.07%        | 90.02%   | ~1 min (CPU)      |  
| **RoBERTa-base**   | **96.22%**    | **96.22%** | 10+ hr (GPU)    |  
| RoBERTa + ICDA+PVI | ~96.0%        | -        | 17+ hr (GPU)      |  

<small>*ICDA+PVI (GPT-2 + PVI filtering) showed no accuracy gains, likely due to generator/model size constraints.*</small>  

---

### Visual Results  

#### 1. **Precision-Recall Curves**  
Micro-averaged precision-recall curves for each model:  

<div style="display: flex; justify-content: space-between;">
  <img src="plots/roberta_pr_curve.png" alt="RoBERTa PR Curve" width="33%" />
  <img src="plots/fasttext_pr_curve.png" alt="FastText PR Curve" width="33%" />
  <img src="plots/nb_pr_curve.png" alt="Naive Bayes PR Curve" width="33%" />
</div>

- **Left**: RoBERTa-base (AP = 0.98)  
- **Middle**: FastText  
- **Right**: Naive Bayes (TF-IDF)   

---

#### 2. **Accuracy and Metrics Bar Plot**  
Comparison of Accuracy, F1, Precision, and Recall for the three main models:  
<div style="text-align: center">
  <img src="plots/metrics_bar_plot.png" alt="Metrics Comparison" style="width: 60%" />
</div>

---

#### 3. **Accuracy Across All Models**  
Bar plot showing test accuracy for all explored models:  
- Classical ML (Naive Bayes, Logistic Regression, SVM, Random Forest, Multilayer Perceptron)  
- FastText  
- RoBERTa-base  

<div style="text-align: center">
  <img src="plots/all_models_accuracy.png" alt="All Models Accuracy" style="width: 60%" />
</div>

---

#### 4. **Class-Level Confusion Matrices**  
Best and worst classes (by recall) for each model:  

| Model           | Best Class (Recall)       | Worst Class (Recall)      |  
|-----------------|------------------------------|-------------------------------|  
| **Naive Bayes** | `whisper_mode` (100%)         | `change_user_name` (13.33%)                   |  
| **FastText**    | `meaning_of_life` (100%)     | `yes` (76.67%)                   |  
| **RoBERTa**     | `cancel` (100%)     | `meal_suggestion` (76.67%)          |  

**Confusion Matrix Examples**:  
<div style="display: flex; justify-content: space-between; gap: 20px; align-items: flex-start">
  <div style="flex: 1; text-align: center">
    <p><strong>Best Class (RoBERTa): </strong><code>cancel</code></p>
    <img src="plots/roberta_best_class.png" alt="RoBERTa Best Class" style="width: 50%; max-width: 400px" />
  </div>
  
  <div style="flex: 1; text-align: center">
    <p><strong>Worst Class (Naive Bayes): </strong><code>change_user_name</code></p>
    <img src="plots/nb_worst_class.png" alt="Naive Bayes Worst Class" style="width: 50%; max-width: 400px" />
  </div>
</div>

---

### Key Observations  
1. **Transformer Superiority**:  
   - RoBERTa-base outperformed classical models by **~15% accuracy** and FastText by **~6%**, demonstrating transformers’ ability to capture contextual nuances.  
   - Achieved near state-of-the-art (SOTA) performance (**96.2%**), comparable to RoBERTa-large (96.8% in literature).  

2. **Data Augmentation Challenges**:  
   - ICDA+PVI required significant computational effort but yielded no gains, highlighting the importance of generator model size (e.g., OPT-66B vs. GPT-2).  

3. **Efficiency Trade-offs**:  
   - Classical models (e.g., Naive Bayes) trained in minutes but plateaued at **81–85% accuracy**.  
   - FastText provided a **6–7% boost** over TF-IDF models with subword embeddings.  

For detailed metrics (precision, recall curves, confusion matrices), see the [project report](CLINC150%20Intent%20Detection%20Final%20Report.docx.pdf).  
