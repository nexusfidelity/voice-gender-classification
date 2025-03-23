# Speaker Gender Classification Report

## 1. Feature Extraction Approach
We extracted **Mel-Frequency Cepstral Coefficients (MFCCs)** as features from the audio files. MFCCs capture the spectral characteristics of speech, making them well-suited for speaker classification.

Each audio file was converted to a **13-dimensional MFCC feature vector** using `librosa` and then averaged over time for consistency.

## 2. Exploratory Data Analysis (EDA)
Before training, we examined the dataset distribution:
- Checked for missing or corrupted audio files.
- Verified the gender label distribution to ensure balance.
- Plotted MFCC distributions to visualize feature separability.

**Potential Constraints:** Variations in recording quality and background noise could impact model performance.

## 3. Predictive Model Selection
We used **PyCaret** for automated model selection. The workflow:
1. Set up a classification task with an 80/20 train-test split.
2. Compared multiple models (SVM, Random Forest, XGBoost, etc.).
3. Selected the best-performing model based on accuracy.
4. Finalized the model and evaluated on test data.

## 4. Model Performance Analysis
### Key Metrics:

![Screenshot 2025-03-23 221824](https://github.com/user-attachments/assets/f9ca09f9-ebda-4e7a-9bce-37262584fcd7)

- **Accuracy:** `XX%` (best model)
- **Precision/Recall:** Balanced for both genders
- **Confusion Matrix:** Showed minor misclassification, primarily in noisy samples

**Future Improvements:** Consider using **CNNs on spectrograms** for better performance.

## 5. Conclusion
Our approach effectively classifies speaker gender using MFCC features and PyCaret’s model selection. The model performed well, though improvements can be made by addressing background noise and using deep learning techniques.
