# Speaker Gender Classification Report

## 1. Data Extraction and Formatting
We extracted `.wav` audio files from **compressed `.tgz` archives** in the VoxForge dataset. Each speaker had their own folder structure, so we:
- **Unpacked** all `.tgz` files.
- **Located and standardized** `.wav` files.
- **Mapped** each file to its corresponding speaker and gender label.
- **Formatted** the dataset for training by ensuring all audio files were converted to a consistent **16kHz, 16-bit** format.

## 2. Feature Extraction Approach
We extracted **Mel-Frequency Cepstral Coefficients (MFCCs)** as features from the audio files. MFCCs capture the spectral characteristics of speech, making them well-suited for speaker classification.

Each audio file was converted to a **13-dimensional MFCC feature vector** using `librosa` and then averaged over time for consistency.

## 3. Why MFCCs?
### **1️⃣ MFCCs Mimic Human Hearing Perception**
- The **mel scale** used in MFCCs mimics how humans perceive sound frequencies.
- Lower frequencies are more important for distinguishing **vocal characteristics**, making MFCCs great for **gender classification**.

### **2️⃣ Capture Speaker-Specific Characteristics**
- MFCCs extract **vocal tract information**, which is key in differentiating **male and female** voices.
- Male voices generally have **lower formant frequencies** and **longer vocal tract lengths**, which MFCCs can capture.

### **3️⃣ Robust Against Background Noise**
- Unlike raw waveforms, MFCCs are more **resistant to variations** in recording conditions.
- Helps in real-world applications where noise levels vary.

### **4️⃣ Efficient & Compact**
- MFCCs reduce the dimensionality of audio signals while retaining important **speech features**.
- This makes model training **faster and more efficient** than using raw waveforms.

### **Why Not Other Features?**

| Feature Type        | Pros | Cons  |
|--------------------|------|------|
| **MFCCs** | Human hearing-based, effective for speech tasks | May lose some time-domain info |
| **Spectrograms** | Retain full frequency details | Higher-dimensional, needs CNNs |
| **Raw Waveforms** | No information loss | Needs more data, complex models (e.g., Wav2Vec) |
| **Chroma Features** | Good for pitch-based tasks | Less relevant for gender classification |

### **Can We Improve MFCCs?**
Yes! We could combine MFCCs with:
- **Spectrograms** (for deep learning models like CNNs).
- **Pitch and energy features** (to improve gender discrimination).

But for **classic ML models (like SVM, Random Forest)**, **MFCCs are the best tradeoff** between accuracy and computational efficiency.

### **Final Verdict: MFCCs Win**
MFCCs provide a **compact, noise-robust, and human-inspired** way to extract **gender-discriminative features** from speech. That’s why they are the gold standard! 🚀

## 4. Exploratory Data Analysis (EDA)
Before training, we examined the dataset distribution:
- Checked for missing or corrupted audio files.
- Verified the gender label distribution to ensure balance.
- Plotted MFCC distributions to visualize feature separability.

**Potential Constraints:** Variations in recording quality and background noise could impact model performance.

## 5. Predictive Model Selection
We used **PyCaret** for automated model selection. The workflow:
1. Set up a classification task with an 80/20 train-test split.
2. Compared multiple models (SVM, Random Forest, XGBoost, etc.).
3. Selected the best-performing model based on accuracy.
4. Finalized the model and evaluated on test data.

## 6. Model Performance Analysis

![Screenshot 2025-03-23 221824](https://github.com/user-attachments/assets/f9ca09f9-ebda-4e7a-9bce-37262584fcd7)

### Key Metrics:
- **Accuracy:** `90.38%` (best model)
- **Precision/Recall:** Balanced for both genders
- **Confusion Matrix:** Showed minor misclassification, primarily in noisy samples

**Future Improvements:** Consider using **CNNs on spectrograms** for better performance.

## 7. Conclusion
The approach effectively classifies speaker gender using MFCC features and PyCaret’s model selection. The model performed well, though improvements can be made by addressing background noise and using deep learning techniques.
