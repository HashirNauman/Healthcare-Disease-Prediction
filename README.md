# Heart Disease Prediction with Random Forest

A streamlined pipeline for predicting heart disease using the UCI Cleveland dataset and a Random Forest classifier.

---

## 🚀 Features

* **Dataset**: UCI Heart Disease (Cleveland)
* **Preprocessing**: Handles missing values, encodes categoricals, standardizes numerical features
* **Model**: Random Forest with hyperparameter tuning via GridSearchCV
* **Evaluation**: Accuracy, classification report, and confusion matrix
* **Persistence**: Saves the best model (`.pkl`) for later inference

---

## 📦 Requirements

* Python 3.7+
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `joblib`

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 📝 Usage

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/heart-disease-rf.git
   cd heart-disease-rf
   ```

2. **Run the script**

   ```bash
   python main.py
   ```

   * Downloads and cleans data
   * Performs train/test split (80/20)
   * Tunes hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`)
   * Trains best model and evaluates on test set
   * Displays accuracy (\~0.8833), classification report, and confusion matrix plot
   * Saves model at `models/heart_disease_model.pkl`

---

## 📊 Results

* **Best Hyperparameters**: `n_estimators=200`, `max_depth=10`, `min_samples_split=5`
* **Test Accuracy**: 0.8833
* **Classification Report**:

  ```
                 precision    recall  f1-score   support

          0       0.91      0.89      0.90        36
          1       0.84      0.88      0.86        24

      accuracy                           0.88        60
     macro avg       0.88      0.88      0.88        60
  ```

weighted avg       0.88      0.88      0.88        60

````

---

## 📂 File Structure

```plaintext
├── Main.py                       # Data loading, preprocessing, training
├── models/                       # Saved model directory
│   └── heart_disease_model.pkl   # Best Random Forest model
└── requirements.txt             # Project dependencies
````

---

## 🔧 Customization

* **Parameter Grid**: Adjust `param_grid` in `tune_hyperparameters()`
* **Train/Test Split**: Modify `test_size` and `random_state` in `train_test_split`
* **Model**: Swap `RandomForestClassifier` for other estimators (e.g., XGBoost)

---

## 📄 License

Licensed under MIT. See [LICENSE](LICENSE) for details.

---

*Enjoy exploring heart disease prediction!*
