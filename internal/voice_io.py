import speech_recognition as sr

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("You can now speak.")
        try:
            audio = recognizer.listen(source)
            print("Processing your speech...")
            user_input = recognizer.recognize_google(audio)
            print(f"You said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
        except sr.RequestError as e:
            print(f"Request error: {e}")
        return None
    
# voice_io.py
import pyttsx3
import string

def speak(text):
    if not isinstance(text, str):
        return  # Bail out if not a string

    if not text.strip():
        return  # Bail out if empty/whitespace

    engine = pyttsx3.init()

    # Clean out weird characters that might break SAPI
    printable = set(string.printable)
    safe_text = "".join(ch if ch in printable else " " for ch in text)
    safe_text = safe_text.replace("\n", " ").replace("\r", " ")

    engine.say(safe_text)
    engine.runAndWait()
