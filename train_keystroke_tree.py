# keystroke_analysis.py

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt

def load_processed_data(directory='processed_data'):
    """
    Load processed keystroke data from JSON files.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('_processed.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                attempt_data = json.load(f)
                data.append(attempt_data)
    return data

def extract_features(attempt_data):
    """
    Extract features from processed keystroke data.
    """
    processed_keystrokes = attempt_data['processed_data']

    # Hold times
    hold_times = [ks['hold_time'] for ks in processed_keystrokes if ks['hold_time'] is not None]

    # Inter-key intervals (IKI)
    inter_key_intervals = [ks['inter_key_interval'] for ks in processed_keystrokes if ks['inter_key_interval'] is not None]

    # Flight times: time between releasing one key and pressing the next
    flight_times = inter_key_intervals  # Assuming inter_key_interval represents flight time

    # Error rate: count of backspaces in processed keystrokes
    backspace_count = sum(1 for ks in processed_keystrokes if ks['key'] == '\b')

    # Typing speed: total number of characters divided by total typing duration
    # Since 'typed_text' is not available, we can estimate total_chars from the number of keystrokes
    total_chars = len([ks for ks in processed_keystrokes if ks['key'] != '\b'])
    total_time = sum(hold_times) + sum(inter_key_intervals)
    typing_speed_cps = total_chars / total_time if total_time > 0 else 0  # Characters per second

    # Statistical measures for hold times
    hold_time_series = pd.Series(hold_times) if hold_times else pd.Series([0])
    hold_time_skewness = hold_time_series.skew()
    hold_time_kurtosis = hold_time_series.kurtosis()

    # Statistical measures for inter-key intervals
    inter_key_interval_series = pd.Series(inter_key_intervals) if inter_key_intervals else pd.Series([0])
    inter_key_interval_skewness = inter_key_interval_series.skew()
    inter_key_interval_kurtosis = inter_key_interval_series.kurtosis()

    # Key-specific hold times
    vowels = {'a', 'e', 'i', 'o', 'u'}
    vowel_hold_times = [
        ks['hold_time'] for ks in processed_keystrokes
        if ks['key'].lower() in vowels and ks['hold_time'] is not None
    ]
    consonant_hold_times = [
        ks['hold_time'] for ks in processed_keystrokes
        if ks['key'].isalpha() and ks['key'].lower() not in vowels and ks['hold_time'] is not None
    ]

    features = {
        'name': attempt_data['name'],
        'attempt': attempt_data['attempt'],

        # Basic hold time features
        'hold_time_mean': hold_time_series.mean(),
        'hold_time_std': hold_time_series.std(),
        'hold_time_median': hold_time_series.median(),

        # Basic inter-key interval features
        'inter_key_interval_mean': inter_key_interval_series.mean(),
        'inter_key_interval_std': inter_key_interval_series.std(),
        'inter_key_interval_median': inter_key_interval_series.median(),

        # Flight time features
        'flight_time_mean': inter_key_interval_series.mean(),
        'flight_time_std': inter_key_interval_series.std(),
        'flight_time_median': inter_key_interval_series.median(),

        # Error rate
        'error_count': backspace_count,

        # Typing speed
        'typing_speed_cps': typing_speed_cps,

        # Statistical measures
        'hold_time_skewness': hold_time_skewness,
        'hold_time_kurtosis': hold_time_kurtosis,
        'inter_key_interval_skewness': inter_key_interval_skewness,
        'inter_key_interval_kurtosis': inter_key_interval_kurtosis,

        # Key-specific hold times
        'vowel_hold_time_mean': np.mean(vowel_hold_times) if vowel_hold_times else 0,
        'consonant_hold_time_mean': np.mean(consonant_hold_times) if consonant_hold_times else 0,
    }

    return features

def get_decision_path(model, feature_names, sample):
    """
    Get the decision path in the decision tree for a given sample.
    """
    node_indicator = model.decision_path(sample)
    leave_id = model.apply(sample)
    feature_index = model.tree_.feature
    threshold = model.tree_.threshold

    sample_id = 0  # Since we have only one sample
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

    explanation = []
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            explanation.append("Decision: Reached leaf node {}.".format(node_id))
            continue

        if sample.iloc[0, feature_index[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        explanation.append(
            "Decision node {}: ({} = {:.4f}) {} {:.4f} (threshold {:.4f})".format(
                node_id,
                feature_names[feature_index[node_id]],
                sample.iloc[0, feature_index[node_id]],
                threshold_sign,
                threshold[node_id],
                threshold[node_id]
            )
        )

    return "\n".join(explanation)

def main():
    # Load the processed data
    processed_data = load_processed_data()
    print(processed_data)
    if not processed_data:
        print("No processed data files found in the 'processed_data' directory.")
        return

    # Extract features for all attempts
    feature_list = [extract_features(attempt) for attempt in processed_data]

    # Create a DataFrame
    df = pd.DataFrame(feature_list)
    print(df.head())
    df.to_csv('out.csv')

    # Encode the user names into labels
    label_encoder = LabelEncoder()
    df['user_id'] = label_encoder.fit_transform(df['name'])

    # Remove classes with fewer than 2 samples
    class_counts = df['user_id'].value_counts()
    sufficient_classes = class_counts[class_counts >= 2].index
    df = df[df['user_id'].isin(sufficient_classes)]

    # Check if there are enough classes left
    if df['user_id'].nunique() < 2:
        print("Not enough classes with sufficient samples for training.")
        return

    # Define feature columns
    feature_columns = [
        'hold_time_mean', 'hold_time_std', 'hold_time_median',
        'inter_key_interval_mean', 'inter_key_interval_std', 'inter_key_interval_median',
        'flight_time_mean', 'flight_time_std', 'flight_time_median',
        'error_count', 'typing_speed_cps',
        'hold_time_skewness', 'hold_time_kurtosis',
        'inter_key_interval_skewness', 'inter_key_interval_kurtosis',
        'vowel_hold_time_mean', 'consonant_hold_time_mean'
    ]

    # Prepare feature matrix X and target vector y
    X = df[feature_columns]
    y = df['user_id']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)  # Use DataFrame with feature names

    # Initialize the Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Adjust n_splits based on the minimum number of samples per class
    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class < 2:
        print("Not enough samples per class to perform StratifiedKFold with n_splits=2.")
        return
    n_splits = min(5, min_samples_per_class)  # You can adjust the maximum number of splits

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Suppress warnings about undefined metrics
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Stratified K-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf)
    print("Cross-validation scores:", cv_scores)
    print("Average CV score: {:.2f}%".format(cv_scores.mean() * 100))

    # Classification Report using cross-validated predictions
    y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.inverse_transform(sorted(y.unique())), zero_division=0))

    # Fit the model on the full dataset
    model.fit(X_scaled, y)

    # Visualize the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_columns, class_names=label_encoder.inverse_transform(model.classes_), filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

    # Explain a sample prediction
    sample_index = 0  # Change this to select a different sample
    sample = X_scaled.iloc[[sample_index]]  # Use DataFrame with feature names
    actual_user = label_encoder.inverse_transform([y.iloc[sample_index]])[0]
    predicted_user = label_encoder.inverse_transform([model.predict(sample)[0]])[0]

    print("Actual User:", actual_user)
    print("Predicted User:", predicted_user)

    explanation = get_decision_path(model, feature_columns, sample)
    print("\nExplanation of the decision path:")
    print(explanation)

    # Save the model and scaler for future use
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/keystroke_decision_tree_model.joblib')
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')

    print("\nModel, scaler, and label encoder saved in the 'models' directory.")

if __name__ == "__main__":
    main()
