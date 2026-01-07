from pynput import keyboard
import time
import json

# Initialize an empty list to store keystroke events
keystroke_data = []
typed_text = []  # List to keep track of typed characters

# Event handler for key press
def on_press(key):
    try:
        # Handle printable characters
        if hasattr(key, 'char') and key.char is not None:
            typed_text.append(key.char)
            print(key.char, end='', flush=True)  # Print the character to the terminal
        # Handle the space key
        elif key == keyboard.Key.space:
            typed_text.append(' ')
            print(' ', end='', flush=True)  # Print a space to the terminal

        # Record key press time and details
        keystroke_data.append({
            'key': key.char if hasattr(key, 'char') else str(key),
            'event': 'press',
            'time': time.time()
        })
    except AttributeError:
        # Handle other special keys like Shift, Ctrl, etc.
        keystroke_data.append({
            'key': str(key),
            'event': 'press',
            'time': time.time()
        })

# Event handler for key release
def on_release(key):
    # Record key release time and details
    keystroke_data.append({
        'key': key.char if hasattr(key, 'char') else str(key),
        'event': 'release',
        'time': time.time()
    })
    # Stop the listener when 'Esc' key is pressed
    if key == keyboard.Key.esc:
        print("\nLogging stopped.")
        return False

# Main function to start the listener
def main():
    print("Start typing... (Press 'Esc' to stop)")
    # Start listening to keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    print("\n\nTyped Text:", ''.join(typed_text))

    # Save raw keystroke data to JSON
    with open("keystroke_raw_data.json", "w") as f:
        json.dump(keystroke_data, f, indent=4)

    print("Keystroke data saved to 'keystroke_raw_data.json'.")

# Run the program
if __name__ == "__main__":
    main()
