import sys
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QPushButton
)
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence

def extract_features_from_attempt(processed_keystrokes):
    """
    Extract features from processed keystroke data.
    """
    # Hold times
    hold_times = [ks['hold_time'] for ks in processed_keystrokes if ks['hold_time'] is not None]

    # Inter-key intervals (IKI)
    inter_key_intervals = [ks['inter_key_interval'] for ks in processed_keystrokes if ks['inter_key_interval'] is not None]

    # Flight times: time between releasing one key and pressing the next
    flight_times = inter_key_intervals  # Assuming inter_key_interval represents flight time

    # Error rate: count of backspaces in processed keystrokes
    backspace_count = sum(1 for ks in processed_keystrokes if ks['key'] == '\b')

    # Typing speed: total number of characters divided by total typing duration
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

def set_font_size(widget, size):
    font = widget.font()
    font.setPointSize(size)
    widget.setFont(font)

class TypingTest(QWidget):
    def __init__(self):
        super().__init__()
        self.keystroke_data = []
        self.given_name = ''
        self.attempt = 0  # To keep track of attempts
        self.keystroke_data_all_attempts = []
        self.initUI()

        # Load the trained model components
        self.load_model_components()

    def load_model_components(self):
        try:
            self.model = joblib.load('models/keystroke_decision_tree_model.joblib')
            self.scaler = joblib.load('models/feature_scaler.joblib')
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            # Define feature columns (must match the training script)
            self.feature_columns = [
                'hold_time_mean', 'hold_time_std', 'hold_time_median',
                'inter_key_interval_mean', 'inter_key_interval_std', 'inter_key_interval_median',
                'flight_time_mean', 'flight_time_std', 'flight_time_median',
                'error_count', 'typing_speed_cps',
                'hold_time_skewness', 'hold_time_kurtosis',
                'inter_key_interval_skewness', 'inter_key_interval_kurtosis',
                'vowel_hold_time_mean', 'consonant_hold_time_mean'
            ]
        except Exception as e:
            print("Error loading model components:", e)
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Typing Test')

        # Make the application full-screen
        self.showFullScreen()

        # Set the background color
        self.setStyleSheet("background-color: #2E3440;")  # Dark background

        # Create or clear the main layout
        if hasattr(self, 'main_layout'):
            # Clear the layout if it exists
            self.clearLayout(self.main_layout)
        else:
            self.main_layout = QVBoxLayout()
            self.setLayout(self.main_layout)

        # Name input
        self.label = QLabel('Enter your full name:')
        set_font_size(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #D8DEE9;")  # Light text color

        self.name_input = QLineEdit()
        self.name_input.returnPressed.connect(self.startTest)
        set_font_size(self.name_input, 24)
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setStyleSheet("""
            QLineEdit {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 2px solid #4C566A;
                border-radius: 10px;
                padding: 10px;
            }
            QLineEdit:focus {
                border: 2px solid #88C0D0;
            }
        """)

        self.main_layout.addStretch()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.name_input)
        self.main_layout.addStretch()

        self.name_input.setFocus()
        self.show()

    def startTest(self):
        self.given_name = self.name_input.text().strip()
        if not self.given_name:
            self.label.setText('Please enter a valid name:')
            return

        # Remove the name input widgets
        self.main_layout.removeWidget(self.label)
        self.label.deleteLater()
        self.main_layout.removeWidget(self.name_input)
        self.name_input.deleteLater()

        # Prepare the required sentence with capital letters and punctuation
        self.required_sentence = (
            f"If a dog chews shoes, whose shoes does he choose."
        )
        self.attempt = 1
        self.total_attempts = 1  # Set total attempts to 1 for testing
        self.keystroke_data_all_attempts = []  # List to store data for all attempts

        self.showInstruction()

    def showInstruction(self):
        # Clear any existing widgets
        self.clearLayout(self.main_layout)

        self.instructions = QLabel(
            f'Type the following sentence ({self.attempt}/{self.total_attempts}):\n\n'
            f'{self.required_sentence}'
        )
        set_font_size(self.instructions, 24)
        self.instructions.setAlignment(Qt.AlignCenter)
        self.instructions.setStyleSheet("color: #D8DEE9;")  # Light text color

        self.text_edit = QTextEdit()
        self.text_edit.installEventFilter(self)  # To capture key events
        set_font_size(self.text_edit, 24)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 2px solid #4C566A;
                border-radius: 10px;
                padding: 10px;
            }
            QTextEdit:focus {
                border: 2px solid #88C0D0;
            }
        """)
        self.text_edit.setFocus()

        self.main_layout.addStretch()
        self.main_layout.addWidget(self.instructions)
        self.main_layout.addWidget(self.text_edit)
        self.main_layout.addStretch()

    def eventFilter(self, obj, event):
        if obj == self.text_edit:
            if event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
                key = event.key()
                text = event.text()
                timestamp = time.time()
                event_type = 'press' if event.type() == QEvent.KeyPress else 'release'

                # Get a string representation of the key
                if text:
                    key_str = text
                else:
                    key_str = QKeySequence(key).toString()

                # Record keystroke data
                self.keystroke_data.append({
                    'key': key_str,
                    'event': event_type,
                    'time': timestamp
                })

                # Check if the user has completed the sentence after key release
                if event_type == 'release':
                    current_text = self.text_edit.toPlainText()
                    if current_text.strip() == self.required_sentence:
                        # Save the data for this attempt
                        self.saveAttemptData()
                        self.attempt += 1
                        if self.attempt > self.total_attempts:
                            # Finish the test
                            self.finishTest()
                        else:
                            # Start next attempt
                            self.showInstruction()
                # Allow normal processing
                return False
        return super().eventFilter(obj, event)

    def saveAttemptData(self):
        # Process data
        processed_data = self.processKeystrokeData(self.keystroke_data)

        # Save the data for this attempt
        attempt_data = {
            'name': self.given_name,
            'attempt': self.attempt,
            'typed_text': self.text_edit.toPlainText(),
            'keystroke_data': self.keystroke_data.copy(),
            'processed_data': processed_data  # Include processed data
        }
        self.keystroke_data_all_attempts.append(attempt_data)

        # File-safe name by replacing spaces with underscores
        safe_name = self.given_name.replace(' ', '_')

        # Ensure directories exist
        os.makedirs('test/raw_data', exist_ok=True)
        os.makedirs('test/processed_data', exist_ok=True)

        # Save raw data
        raw_filename = f"test/raw_data/{safe_name}_attempt_{self.attempt}_raw.json"
        with open(raw_filename, "w") as f:
            json.dump(attempt_data, f, indent=4)

        # Save processed data
        processed_attempt_data = {
            'name': self.given_name,
            'attempt': self.attempt,
            'processed_data': processed_data,
            'keystroke_data': self.keystroke_data.copy(),
            'typed_text': self.text_edit.toPlainText()
        }
        processed_filename = f"test/processed_data/{safe_name}_attempt_{self.attempt}_processed.json"
        with open(processed_filename, "w") as f:
            json.dump(processed_attempt_data, f, indent=4)

        # Reset keystroke data for the next attempt
        self.keystroke_data = []

    def processKeystrokeData(self, keystroke_data):
        # Process the raw keystroke data to compute key hold times and inter-key intervals
        processed_data = []
        key_press_times = {}
        last_release_time = None

        for event in keystroke_data:
            key = event['key']
            event_type = event['event']
            time_stamp = event['time']

            if event_type == 'press':
                key_press_times[key] = time_stamp
            elif event_type == 'release':
                if key in key_press_times:
                    hold_time = time_stamp - key_press_times[key]
                    inter_key_interval = None
                    if last_release_time is not None:
                        inter_key_interval = key_press_times[key] - last_release_time
                    processed_data.append({
                        'key': key,
                        'hold_time': hold_time,
                        'inter_key_interval': inter_key_interval
                    })
                    last_release_time = time_stamp
                    del key_press_times[key]

        return processed_data

    def finishTest(self):
        # Process the collected data to extract features
        # Use the data from the last attempt
        last_attempt_data = self.keystroke_data_all_attempts[-1]
        processed_keystrokes = last_attempt_data['processed_data']
        features = extract_features_from_attempt(processed_keystrokes)
        feature_values = [features[col] for col in self.feature_columns]

        # Normalize the features
        features_scaled = self.scaler.transform([feature_values])

        # Predict the user
        user_id_pred = self.model.predict(features_scaled)
        user_name_pred = self.label_encoder.inverse_transform(user_id_pred)
        predicted_user = user_name_pred[0]

        # Provide explanation
        sample = pd.DataFrame(features_scaled, columns=self.feature_columns)
        explanation = self.get_decision_path(sample)

        # Clear the layout
        self.clearLayout(self.main_layout)

        # Show completion message
        self.completion_label = QLabel(f"Authentication Result:\nPredicted User: {predicted_user}")
        set_font_size(self.completion_label, 24)
        self.completion_label.setAlignment(Qt.AlignCenter)
        self.completion_label.setStyleSheet("color: #D8DEE9;")  # Light text color
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.completion_label)

        # Show the explanation
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlainText(f"Explanation:\n{explanation}")
        set_font_size(self.explanation_text, 16)
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #3B4252;
                color: #ECEFF4;
                border: 2px solid #4C566A;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.main_layout.addWidget(self.explanation_text)

        # "Get Another Sample" button
        self.restart_button = QPushButton("Get Another Sample")
        self.restart_button.clicked.connect(self.restartTest)
        set_font_size(self.restart_button, 18)
        self.restart_button.setFixedWidth(300)
        self.restart_button.setFixedHeight(60)
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #5E81AC;
                color: #ECEFF4;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
        """)
        self.main_layout.addWidget(self.restart_button, alignment=Qt.AlignCenter)
        self.main_layout.addStretch()

        # Ensure the restart button has focus
        self.restart_button.setFocus()

    def get_decision_path(self, sample):
        """
        Get the decision path in the decision tree for a given sample.
        """
        node_indicator = self.model.decision_path(sample)
        leave_id = self.model.apply(sample)
        feature_index = self.model.tree_.feature
        threshold = self.model.tree_.threshold

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
                    self.feature_columns[feature_index[node_id]],
                    sample.iloc[0, feature_index[node_id]],
                    threshold_sign,
                    threshold[node_id],
                    threshold[node_id]
                )
            )

        return "\n".join(explanation)

    def restartTest(self):
        # Reset variables
        self.keystroke_data = []
        self.given_name = ''
        self.attempt = 0
        self.keystroke_data_all_attempts = []

        # Re-initialize UI
        self.initUI()

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

def main():
    app = QApplication(sys.argv)
    ex = TypingTest()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
