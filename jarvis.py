#!/usr/bin/env python3
"""
Jarvis - A voice-activated AI personal assistant powered by Claude.

Setup (Windows 11):
    1. Install Python 3.10+ from https://python.org (check "Add to PATH")
    2. pip install -r requirements.txt
    3. Set your API key:
           set ANTHROPIC_API_KEY=your-api-key-here
       Or permanently via System Properties > Environment Variables.
    4. Install a British male voice (if not already present):
           Settings > Time & Language > Speech > Manage voices > Add voices
           Select "English (United Kingdom)" to install voices like
           "Microsoft George" (male, British).

Setup (Linux):
    1. sudo apt-get install portaudio19-dev espeak-ng python3-pyaudio
    2. pip install -r requirements.txt
    3. export ANTHROPIC_API_KEY='your-api-key-here'

Usage:
    python jarvis.py
"""

import json
import math
import os
import platform
import re
import struct
import sys
from datetime import datetime
from pathlib import Path

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
    HISTORY_DIR = Path.home() / ".jarvis"
    HISTORY_FILE = HISTORY_DIR / "history.json"
    MAX_HISTORY_MESSAGES = 100  # rolling window sent to Claude
    MAX_STORED_SESSIONS = 50   # sessions kept on disk

    def __init__(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            if platform.system() == "Windows":
                print("Set it with: set ANTHROPIC_API_KEY=your-key-here")
            else:
                print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)

        self.client = anthropic.Anthropic()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.state = "ACTIVE"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._load_history()
        self._setup_voice()
        self._calibrate_microphone()

    def _setup_voice(self):
        """Configure text-to-speech for a British male voice."""
        voices = self.engine.getProperty("voices")
        system = platform.system()

        best = None
        fallback = None

        if system == "Windows":
            # Windows SAPI5 voices — look for British English male
            for voice in voices:
                vid = voice.id.lower()
                vname = (voice.name or "").lower()
                is_british = "en-gb" in vid or "en_gb" in vid
                is_male = "george" in vid or "george" in vname
                if is_british and is_male:
                    best = voice
                    break
                if is_british and best is None:
                    best = voice
                if not fallback and ("david" in vname or "en-us" in vid):
                    fallback = voice
        else:
            # Linux/macOS espeak voices
            for voice in voices:
                vid = voice.id.lower()
                if any(
                    tag in vid
                    for tag in ["english_rp", "english-gb", "en-gb", "en_gb"]
                ):
                    best = voice
                    break
                if "english" in vid and fallback is None:
                    fallback = voice

        selected = best or fallback
        if selected:
            self._voice_id = selected.id
            print(f"Voice selected: {selected.name or selected.id}")
        else:
            self._voice_id = None
            print("Warning: No British English voice found. Using system default.")
            if system == "Windows":
                print(
                    "Tip: Install British voices via Settings > Time & Language > "
                    "Speech > Manage voices > Add voices > English (United Kingdom)"
                )

        # Apply settings to the initial engine
        self._apply_voice_settings(self.engine)

    def _apply_voice_settings(self, engine):
        """Apply voice, rate, and volume to the given engine instance."""
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

    def _calibrate_microphone(self):
        """Adjust recognizer sensitivity for ambient noise."""
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone calibrated.")

    def _load_history(self):
        """Load conversation history from disk."""
        self.all_sessions = []
        self.conversation_history = []

        if self.HISTORY_FILE.exists():
            try:
                data = json.loads(self.HISTORY_FILE.read_text(encoding="utf-8"))
                self.all_sessions = data.get("sessions", [])
            except (json.JSONDecodeError, OSError):
                self.all_sessions = []

        # Build a flat message list from recent sessions for Claude context
        self.prior_messages = []
        for session in self.all_sessions:
            for msg in session.get("messages", []):
                self.prior_messages.append(msg)

        # Keep only the tail to stay within token limits
        if len(self.prior_messages) > self.MAX_HISTORY_MESSAGES:
            self.prior_messages = self.prior_messages[-self.MAX_HISTORY_MESSAGES :]

        if self.all_sessions:
            print(
                f"Loaded history: {len(self.all_sessions)} prior session(s), "
                f"{len(self.prior_messages)} messages in context."
            )

    def _save_history(self):
        """Persist conversation history to disk."""
        self.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        # Update or append the current session
        current = {
            "session_id": self.session_id,
            "date": datetime.now().isoformat(),
            "messages": self.conversation_history,
        }

        # Replace current session entry if it already exists
        updated = False
        for i, s in enumerate(self.all_sessions):
            if s.get("session_id") == self.session_id:
                self.all_sessions[i] = current
                updated = True
                break
        if not updated:
            self.all_sessions.append(current)

        # Trim old sessions
        if len(self.all_sessions) > self.MAX_STORED_SESSIONS:
            self.all_sessions = self.all_sessions[-self.MAX_STORED_SESSIONS :]

        data = {"sessions": self.all_sessions}
        self.HISTORY_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def speak(self, text):
        """Speak the given text aloud and print it."""
        print(f"Jarvis: {text}")
        # Reinitialize the engine each call to avoid SAPI5 state corruption
        # after mixing with PyAudio (known pyttsx3 issue on Windows).
        engine = pyttsx3.init()
        self._apply_voice_settings(engine)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

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

        # Combine prior session history with current session for full context
        full_context = self.prior_messages + self.conversation_history
        # Trim to stay within limits
        if len(full_context) > self.MAX_HISTORY_MESSAGES:
            full_context = full_context[-self.MAX_HISTORY_MESSAGES :]

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=(
                    "You are Jarvis, an intelligent and helpful personal assistant. "
                    "You speak in a polished, articulate manner with dry wit, similar to "
                    "a refined British butler. Keep your responses concise and conversational "
                    "since they will be read aloud. Avoid markdown formatting, bullet points, "
                    "code blocks, or any visual formatting. Use plain spoken English only.\n\n"
                    "You have access to conversation history from previous sessions. "
                    "When relevant, reference things the user has discussed before to "
                    "provide continuity and a personalised experience. If the user asks "
                    "about something you discussed previously, draw on that context."
                ),
                messages=full_context,
            )
            reply = response.content[0].text
            self.conversation_history.append({"role": "assistant", "content": reply})
            self._save_history()
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
