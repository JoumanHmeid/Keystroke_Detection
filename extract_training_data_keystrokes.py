import sys
import os
import time
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QPushButton
)
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent, QFont, QKeySequence


class TypingTest(QWidget):
    def __init__(self):
        super().__init__()
        self.keystroke_data = []
        self.given_name = ''
        self.attempt = 0  # To keep track of attempts
        self.initUI()

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
            f"Hello world, "
            "I scream, you scream, we all scream for ice cream."
        )
        self.attempt = 1
        self.keystroke_data_all_attempts = []  # List to store data for all attempts

        self.showInstruction()

    def showInstruction(self):
        # Clear any existing widgets
        self.clearLayout(self.main_layout)

        self.instructions = QLabel(
            f'Type the following sentence ({self.attempt}/3):\n\n'
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
                        if self.attempt > 3:
                            # Finish the test
                            self.finishTest()
                        else:
                            # Start next attempt
                            self.showInstruction()
                # Allow normal processing
                return False
        return super().eventFilter(obj, event)

    def saveAttemptData(self):
        # Save the data for this attempt
        attempt_data = {
            'name': self.given_name,
            'attempt': self.attempt,
            'typed_text': self.text_edit.toPlainText(),
            'keystroke_data': self.keystroke_data.copy()
        }
        self.keystroke_data_all_attempts.append(attempt_data)

        # File-safe name by replacing spaces with underscores
        safe_name = self.given_name.replace(' ', '_')

        # Ensure directories exist
        os.makedirs('raw_data', exist_ok=True)
        os.makedirs('processed_data', exist_ok=True)

        # Save raw data
        raw_filename = f"raw_data/{safe_name}_attempt_{self.attempt}_raw.json"
        with open(raw_filename, "w") as f:
            json.dump(attempt_data, f, indent=4)

        # Process data and save
        processed_data = self.processKeystrokeData(self.keystroke_data)
        processed_attempt_data = {
            'name': self.given_name,
            'attempt': self.attempt,
            'processed_data': processed_data
        }
        processed_filename = f"processed_data/{safe_name}_attempt_{self.attempt}_processed.json"
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
        # Clear the layout
        self.clearLayout(self.main_layout)

        # Show completion message
        self.completion_label = QLabel("Thank you for completing the typing test.")
        set_font_size(self.completion_label, 24)
        self.completion_label.setAlignment(Qt.AlignCenter)
        self.completion_label.setStyleSheet("color: #D8DEE9;")  # Light text color
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.completion_label)

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


def set_font_size(widget, size):
    font = widget.font()
    font.setPointSize(size)
    widget.setFont(font)


def main():
    app = QApplication(sys.argv)
    ex = TypingTest()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
