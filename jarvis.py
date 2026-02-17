#!/usr/bin/env python3
"""
Jarvis - A voice-activated AI personal assistant powered by Claude.

System dependencies (install before running):
    Ubuntu/Debian:
        sudo apt-get install portaudio19-dev espeak-ng python3-pyaudio
    Fedora:
        sudo dnf install portaudio-devel espeak-ng python3-pyaudio
    macOS:
        brew install portaudio espeak

Python dependencies:
    pip install -r requirements.txt

Environment:
    export ANTHROPIC_API_KEY='your-api-key-here'

Usage:
    python jarvis.py
"""

import math
import os
import re
import struct
import sys

import anthropic
import pyaudio
import pyttsx3
import speech_recognition as sr


class Jarvis:
    """Voice-activated AI assistant using Claude."""

    WAKE_PHRASE = "hey jarvis"
    SILENCE_TIMEOUT = 5
    ACTIVE_PHRASE_LIMIT = 30
    WAITING_PHRASE_LIMIT = 10
    BEEP_FREQUENCY = 800
    BEEP_DURATION = 0.15
    SAMPLE_RATE = 44100

    def __init__(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)

        self.client = anthropic.Anthropic()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.conversation_history = []
        self.state = "ACTIVE"

        self._setup_voice()
        self._calibrate_microphone()

    def _setup_voice(self):
        """Configure text-to-speech for a British male voice."""
        voices = self.engine.getProperty("voices")

        # Find the best British English voice available
        best = None
        fallback = None

        for voice in voices:
            vid = voice.id.lower()
            # British English identifiers in espeak/espeak-ng
            if any(
                tag in vid
                for tag in ["english_rp", "english-gb", "en-gb", "en_gb"]
            ):
                best = voice
                break
            # Any English voice as fallback
            if "english" in vid and fallback is None:
                fallback = voice

        selected = best or fallback
        if selected:
            self.engine.setProperty("voice", selected.id)
            print(f"Voice selected: {selected.id}")
        else:
            print("Warning: No British English voice found. Using system default.")

        # Moderate speaking rate for clarity
        self.engine.setProperty("rate", 170)
        self.engine.setProperty("volume", 1.0)

    def _calibrate_microphone(self):
        """Adjust recognizer sensitivity for ambient noise."""
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone calibrated.")

    def speak(self, text):
        """Speak the given text aloud and print it."""
        print(f"Jarvis: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def play_beep(self):
        """Play a short beep tone to indicate Jarvis is listening."""
        p = pyaudio.PyAudio()
        n_samples = int(self.SAMPLE_RATE * self.BEEP_DURATION)
        fade_len = max(1, int(n_samples * 0.1))

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            output=True,
        )

        # Generate a sine wave with fade-in/fade-out to avoid clicks
        buf = bytearray()
        for i in range(n_samples):
            sample = math.sin(2.0 * math.pi * self.BEEP_FREQUENCY * i / self.SAMPLE_RATE)

            # Fade envelope
            if i < fade_len:
                sample *= i / fade_len
            elif i > n_samples - fade_len:
                sample *= (n_samples - i) / fade_len

            buf.extend(struct.pack("<h", int(sample * 16000)))

        stream.write(bytes(buf))
        stream.stop_stream()
        stream.close()
        p.terminate()

    def listen(self, timeout=None, phrase_time_limit=None):
        """Listen for speech and return the recognized text, or None."""
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                return None

        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None

    def ask_claude(self, user_input):
        """Send user input to Claude and return the spoken response."""
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=(
                    "You are Jarvis, an intelligent and helpful personal assistant. "
                    "You speak in a polished, articulate manner with dry wit, similar to "
                    "a refined British butler. Keep your responses concise and conversational "
                    "since they will be read aloud. Avoid markdown formatting, bullet points, "
                    "code blocks, or any visual formatting. Use plain spoken English only."
                ),
                messages=self.conversation_history,
            )
            reply = response.content[0].text
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except anthropic.APIError as e:
            error_msg = "I'm sorry, I encountered an error reaching my systems. Please try again."
            print(f"Claude API error: {e}")
            return error_msg

    @staticmethod
    def _normalize(text):
        """Strip punctuation and normalize case for wake-word matching."""
        return re.sub(r"[^\w\s]", "", text).lower().strip()

    def _extract_command_after_wake(self, text):
        """Return any words following 'hey jarvis' in the utterance, or None."""
        normalized = self._normalize(text)
        idx = normalized.find(self.WAKE_PHRASE)
        if idx == -1:
            return None
        remainder = normalized[idx + len(self.WAKE_PHRASE) :].strip()
        if not remainder:
            return None
        # Return the original-cased portion (approximate offset)
        orig_lower = text.lower()
        orig_idx = orig_lower.find("hey jarvis")
        if orig_idx == -1:
            # Punctuation may have shifted things; just use normalized remainder
            return remainder
        return text[orig_idx + len("hey jarvis") :].strip().strip(",.!?;:")

    def _handle_active(self):
        """ACTIVE state: play beep, listen, respond. Returns False to exit."""
        self.play_beep()
        text = self.listen(
            timeout=self.SILENCE_TIMEOUT,
            phrase_time_limit=self.ACTIVE_PHRASE_LIMIT,
        )

        if text is None:
            print("[No speech detected — entering standby mode]")
            print("[Say 'Hey Jarvis' to resume]")
            self.state = "WAITING"
            return True

        print(f"You: {text}")
        normalized = self._normalize(text)

        # Exit commands
        if normalized in (
            "goodbye",
            "goodbye jarvis",
            "exit",
            "quit",
            "shut down",
            "shutdown",
        ):
            self.speak("Goodbye. Do let me know if you need anything.")
            return False

        # Send to Claude and speak the response
        response = self.ask_claude(text)
        self.speak(response)
        return True

    def _handle_waiting(self):
        """WAITING state: listen for the wake phrase. Returns False to exit."""
        # Listen indefinitely for any speech
        text = self.listen(timeout=None, phrase_time_limit=self.WAITING_PHRASE_LIMIT)

        if text is None:
            return True

        normalized = self._normalize(text)

        if self.WAKE_PHRASE not in normalized:
            return True

        # Wake phrase detected — check for a trailing command
        command = self._extract_command_after_wake(text)

        if command:
            self.speak("Yes, I'm here.")
            self.state = "ACTIVE"
            print(f"You: {command}")
            response = self.ask_claude(command)
            self.speak(response)
        else:
            self.speak("Yes, I'm here.")
            self.state = "ACTIVE"

        return True

    def run(self):
        """Main assistant loop."""
        print()
        print("=" * 50)
        print("  JARVIS — AI Personal Assistant")
        print("  Say 'Hey Jarvis' to wake")
        print("  Say 'Goodbye' to exit")
        print("=" * 50)
        print()

        self.speak("Hello, my name is Jarvis. How can I help you today?")
        self.state = "ACTIVE"

        try:
            while True:
                if self.state == "ACTIVE":
                    if not self._handle_active():
                        break
                elif self.state == "WAITING":
                    if not self._handle_waiting():
                        break
        except KeyboardInterrupt:
            print()
            self.speak("Goodbye.")


def main():
    jarvis = Jarvis()
    jarvis.run()


if __name__ == "__main__":
    main()
