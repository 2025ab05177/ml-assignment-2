### Machine Learning Assignment 2 (Classification + Streamlit Deployment)

### a. Problem Statement
The objective of this project is to implement and evaluate six different machine learning classification models on a single dataset and deploy an interactive Streamlit web application. The models are used to recognize human activities (such as walking, sitting, standing, etc.) from smartphone sensor signals. The work demonstrates an end-to-end machine learning workflow including dataset preparation, preprocessing, model training, evaluation using multiple metrics, and deployment.

This project satisfies the assignment requirements:
- Dataset has **≥ 12 features** and **≥ 500 instances**
- Implemented **6 ML classification models**
- Computed **6 evaluation metrics** for each model
- Built and deployed an interactive **Streamlit** web application
- Shared **GitHub repository** link and **Streamlit cloud app** link (to be filled below)

### b. Dataset Description
**Dataset Name:** Human Activity Recognition Using Smartphones (HAR)

**Dataset Source:** UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

**Dataset Summary**
- **Task Type:** Multi-class classification
- **Number of Instances:** 10,299
- **Number of Features:** 561 (meets minimum required feature size ≥ 12)
- **Target / Label Column Used in This Project:** `Activity`
- **Number of Classes:** 6
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING

**Short Description**
The dataset contains features extracted from accelerometer and gyroscope readings of a waist-mounted smartphone. Each record represents a fixed time-window of sensor signals converted into a 561-dimensional feature vector with an associated activity label.

### c. Models Implemented
The following 6 classification models were implemented:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (kNN)**
4. **Naive Bayes (GaussianNB)**
5. **Random Forest (Ensemble)**
6. **XGBoost (Ensemble)**

### d. Evaluation Metrics Used
For each model, the following evaluation metrics were computed (as required):
1. **Accuracy**
2. **AUC Score** (multi-class ROC AUC using One-vs-Rest)
3. **Precision** (weighted)
4. **Recall** (weighted)
5. **F1 Score** (weighted)
6. **MCC (Matthews Correlation Coefficient)**

### e. Model Performance Comparison Table (Artifacts Output)
The following results are taken directly from the generated `artifacts/metrics.csv` output:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9369 | 0.9619 | 0.9356 | 0.9364 | 0.9359 | 0.9241 |
| KNN | 0.9631 | 0.9986 | 0.9665 | 0.9635 | 0.9642 | 0.9560 |
| Logistic Regression | 0.9859 | 0.9995 | 0.9866 | 0.9864 | 0.9865 | 0.9831 |
| Naive Bayes | 0.7243 | 0.9606 | 0.7849 | 0.7338 | 0.7180 | 0.6877 |
| Random Forest | 0.9820 | 0.9996 | 0.9817 | 0.9817 | 0.9816 | 0.9784 |
| XGBoost | **0.9942** | **0.9999** | **0.9939** | **0.9942** | **0.9940** | **0.9930** |

### f. Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Decision Tree | Performed well but lower than ensemble models. A single tree can overfit with many features and may not generalize as strongly. |
| KNN | Strong performance after scaling (distance-based learning works well). Can be slower during prediction for large datasets. |
| Logistic Regression | Excellent performance (0.9859 accuracy). Indicates features are highly informative and largely separable using a linear boundary after scaling. |
| Naive Bayes | Lowest performance (0.7243). The independence assumption does not hold well because HAR features are correlated. |
| Random Forest | Very high performance (0.9820). Ensemble averaging reduces variance and improves generalization over a single decision tree. |
| XGBoost | Best overall model (0.9942). Boosting improves accuracy by sequentially correcting errors and optimizing loss effectively. |

### g. Streamlit Web Application (Assignment Requirements)
The Streamlit app provides the following features:
1. **Dataset Upload Option (CSV):**
   - Upload a CSV file with **561 feature columns**
   - Optional label support: include `Activity` column to evaluate metrics
2. **Model Selection Dropdown:**
   - Select from 6 trained models
3. **Evaluation Metrics Display:**
   - Displays Accuracy, Precision, Recall, F1, MCC, and AUC (when labels exist)
4. **Confusion Matrix / Classification Report:**
   - Confusion matrix and classification report are shown when `Activity` labels are present in the uploaded CSV

**What to upload in the app?**
- Upload `har_test.csv` (recommended) because it contains 561 features + `Activity` labels.
- If you upload a file without `Activity`, the app will still predict, but it cannot compute evaluation metrics.

### h. Project Folder Structure
A typical structure for this project is:
- `app.py` : Streamlit UI
- `scripts/make_har_csv.py` : Converts HAR raw dataset to CSV files
- `model/` : Model training and model builder modules
- `artifacts/` : Saved models and metrics output (`metrics.csv`, encoders, etc.)
- `requirements.txt` : Python dependencies
- `README.md` : Project documentation

### i. Step-by-Step Execution (Local / BITS Virtual Lab / macOS)

#### Step 1: Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Download and unzip HAR dataset
1. Download the dataset zip from UCI:  
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Unzip the dataset into the project folder.
3. Ensure the folder name is exactly:
   - `UCI_HAR_Dataset/`

#### Step 4: Generate CSV files from raw dataset
```bash
python scripts/make_har_csv.py
```

This generates:
- `har_full.csv`
- `har_train.csv`
- `har_test.csv`

#### Step 5: Train all models and generate artifacts
```bash
python -m model.train_models --csv har_full.csv --target Activity
```

Outputs:
- `artifacts/metrics.csv`
- `artifacts/*.joblib` (saved models)
- `artifacts/meta.json`
- `artifacts/label_encoder.joblib`

#### Step 6: Run Streamlit app
```bash
streamlit run app.py
```

#### Step 7: Test inside the Streamlit app
Upload:
- `har_test.csv` (best option, includes `Activity` labels)


### j. Links 
- **GitHub Repository Link:** <https://github.com/2025ab05177/ml-assignment-2>
- **Live Streamlit App Link:** <https://ml-assignment-2-5su8o7ueh2ovvlrojv64pf.streamlit.app/>