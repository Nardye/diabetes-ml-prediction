# Diabetes Hospital Readmission Prediction Using Supervised Machine Learning

## Project overview

This project explores supervised machine learning methods for predicting whether a hospital encounter involving a patient with diabetes is followed by readmission. The goal is to evaluate model performance and identify patterns that may support further research into follow-up care.

**This is an educational portfolio project, not a clinically validated decision-support system.** Its predictions should not be used to make decisions about individual patients.

## Dataset

- Expected local file: `diabetes_clean.csv`
- Target column: `readmission_flag`
- Target coding: `0` = not readmitted; `1` = readmitted
- The notebook describes 10,000 encounters.

The dataset is not bundled automatically with these project files. Confirm that the dataset's license and sharing terms allow publication before uploading it to a public repository. If it cannot be shared, document its source and provide instructions for obtaining it appropriately.

## Project workflow

1. Inspect the data, target distribution, missingness, and duplicate rows.
2. Clean data using documented exclusions.
3. Explore readmission patterns by age, prior utilization, length of stay, diagnosis count, and medication count.
4. Review engineered features and potential leakage.
5. Split data into stratified training and test sets.
6. Fit preprocessing within scikit-learn pipelines.
7. Compare Logistic Regression, class-weighted Logistic Regression, and Random Forest using stratified cross-validation on training data.
8. Tune Random Forest hyperparameters with `GridSearchCV`.
9. Evaluate candidate models on the held-out test set.
10. Review feature importance and run a robustness check excluding encounters coded as expired.

## Repository files

- `Diabetic_prediction_revised (2).ipynb` — exploratory analysis, visualizations, modeling workflow, results, and interpretation.
- `diabetes_model.py` — reusable functions for loading, cleaning, splitting, preprocessing, training, cross-validation, evaluation, tuning, and feature importance.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — excludes temporary files and local dataset files by default.

## Setup

Use Python 3.10 or newer if possible.

```bash
python -m pip install -r requirements.txt
```

Place `diabetes_clean.csv` in the same directory as the notebook and `diabetes_model.py`, if you are permitted to store and use that file locally.

## Run the notebook

1. Open the project folder in Jupyter Notebook or JupyterLab.
2. Open `Diabetic_prediction_revised (2).ipynb`.
3. Run the notebook cells in order.
4. Make sure `diabetes_model.py` and `diabetes_clean.csv` are available in the notebook's working directory.

If the notebook is started from a different folder, change the working directory to the project directory or add that directory to Python's module search path before importing `diabetes_model`.

## Models and evaluation

The notebook compares:
- Logistic Regression
- Balanced Logistic Regression
- Random Forest
- Tuned Random Forest

Metrics include accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation is used for model selection; the held-out test set is intended for final evaluation.

## Interpretation and limitations

- Associations in the exploratory analysis do not establish causation.
- Feature importance describes predictive contribution, not causal effects.
- Performance is moderate in the notebook's earlier run; rerun the notebook and report the actual output rather than treating old values as guaranteed.
- The treatment of encounters ending in death or hospice requires a documented decision based on the target definition and intended prediction population.
- If repeated encounters from the same patient exist and patient identifiers are unavailable, train/test overlap by patient cannot be ruled out.
- Dropping high-missingness lab-result fields may discard information encoded by whether a test was performed.
- External validation, calibration, fairness assessment, and clinically appropriate decision thresholds would be needed before considering clinical use.

## Reproducibility

The model code uses a fixed random seed for splitting and cross-validation. Exact results can still vary with package versions or changes to the dataset. Record the final metrics from your own run in the notebook and update the conclusion accordingly.

## License

Add a license only after deciding how the code and any data may be reused. Do not imply that the dataset is yours if it came from a third party.
