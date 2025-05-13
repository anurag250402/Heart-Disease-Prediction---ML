# ❤️ Heart Disease Prediction

This project uses machine learning to predict the likelihood of heart disease using data from the **Framingham Heart Study**. It includes data preprocessing, visualization, and training a classification model.

---

## 📁 Dataset

We use the `framingham.csv` dataset, which includes features like:

- Age, Sex
- Blood pressure, Cholesterol
- BMI, Smoking status
- Diabetes, Heart rate
- Ten-year risk of coronary heart disease (`TenYearCHD` - target)

---

## 🛠️ How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Heart_Disease_Prediction_.ipynb
   ```

---

## 📦 Requirements

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- statsmodels  

Install them manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

---

## 📊 Project Workflow

### 🔹 Data Preprocessing

```python
import pandas as pd

disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(inplace=True)
```

### 🔹 Feature Scaling

```python
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit_transform(disease_df.drop('TenYearCHD', axis=1))
y = disease_df['TenYearCHD']
```

### 🔹 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
```

### 🔹 Model Training & Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📈 Output Sample

```
              precision    recall  f1-score   support

           0       0.85      0.99      0.92       951
           1       0.61      0.08      0.14       175

    accuracy                           0.85      1126
   macro avg       0.73      0.54      0.53      1126
```

---

## 🚀 Improvements to Try

- Try different models: RandomForest, XGBoost
- Handle class imbalance with SMOTE or undersampling
- Hyperparameter tuning using GridSearchCV

---

## 📄 License

This project is open-source and free to use.

---

## 🙋‍♂️ Author

Made with ❤️ by [Anurag Tripathi](https://github.com/anurag250402)
