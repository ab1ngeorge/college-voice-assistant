import re
import logging
import tempfile
import os
from typing import Tuple, Optional

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Simple heuristic language detector:
    - If text contains Malayalam Unicode characters -> "ml"
    - Else if it contains common Manglish tokens -> "manglish"
    - Else -> "en"
    """

    MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")

    # Common Manglish tokens (extend as needed)
    MANGLISH_TOKENS = {
        "anu", "ethra", "evide", "ennu", "eppo", "aayirikkum", "aakum", "illa",
        "innu", "angane", "entha", "enthaanu", "poyi", "vannu", "njan", "nee",
    }

    def detect_language(self, text: str) -> str:
        if not text:
            return "en"
        # If Malayalam script present
        if self.MALAYALAM_RE.search(text):
            return "ml"
        # Lowercase tokens check for Manglish
        words = re.findall(r"[A-Za-z]+", text.lower())
        manglish_hits = sum(1 for w in words if w in self.MANGLISH_TOKENS)
        # Heuristic: if at least one Manglish token is present, treat as Manglish
        if manglish_hits > 0:
            return "manglish"
        return "en"


class VoiceProcessor:
    """
    Handles Speech-to-Text and Text-to-Speech with simple language detection.
    - speech_to_text accepts sr.AudioData or path via process_audio_file
    - text_to_speech returns the path to the generated mp3 file
    - process_audio_file reads a file path and returns (text, detected_language)
    """

    def __init__(self, recognizer: Optional[sr.Recognizer] = None):
        self.recognizer = recognizer or sr.Recognizer()
        self.language_detector = LanguageDetector()

        # Optional recognizer tuning
        # self.recognizer.energy_threshold = 300
        # self.recognizer.dynamic_energy_threshold = True

    def speech_to_text(self, audio_data: sr.AudioData, language_hint: Optional[str] = None) -> str:
        """
        Convert sr.AudioData to text using Google Web Speech API.
        language_hint: "ml", "manglish", or None/"en"
        Returns recognized text or empty string on failure.
        """
        try:
            if language_hint == "ml":
                lang_code = "ml-IN"
            elif language_hint == "manglish":
                # Manglish uses English recognizer; typical locale en-IN or en-US
                lang_code = "en-IN"
            else:
                lang_code = "en-US"

            # Recognize (this performs network call)
            text = self.recognizer.recognize_google(audio_data, language=lang_code)
            return text
        except sr.UnknownValueError:
            logger.debug("Speech not understood by recognizer.")
            return ""
        except sr.RequestError as e:
            logger.error("Could not request results from Google Speech Recognition service; %s", e)
            return ""

    def text_to_speech(self, text: str, language: str = "en", save_path: Optional[str] = None,
                       play_audio: bool = False) -> str:
        """
        Convert text to speech using gTTS. Returns path to the mp3 file.
        language: "ml", "manglish", or "en"
        - "ml" -> gTTS lang "ml"
        - "manglish" -> use "en" for TTS
        """
        if not text:
            raise ValueError("Text for TTS is empty.")

        if language == "ml":
            lang_code = "ml"
        else:
            # manglish or english -> use English TTS voice
            lang_code = "en"

        # Create a temporary file if save_path not provided
        if not save_path:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            save_path = tmp.name
            tmp.close()  # close so gTTS can write to it on Windows

        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(save_path)
        except Exception as e:
            logger.exception("gTTS failed: %s", e)
            # Clean up partial file if created
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            raise

        if play_audio:
            try:
                audio = AudioSegment.from_file(save_path, format="mp3")
                play(audio)
            except Exception as e:
                logger.warning("Playback failed: %s", e)

        return save_path

    def process_audio_file(self, audio_file_path: str, do_adjust_for_ambient: bool = True) -> Tuple[str, str]:
        """
        Load audio file, run STT, and detect language.
        Returns: (recognized_text, detected_language) where detected_language in {"ml","manglish","en"}
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        with sr.AudioFile(audio_file_path) as source:
            if do_adjust_for_ambient:
                # Listen briefly to adjust energy threshold (optional)
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception as e:
                    # adjust_for_ambient_noise may consume the source; if it fails, we ignore
                    logger.debug("adjust_for_ambient_noise failed: %s", e)
                    pass
                # Need to reopen the file because adjust_for_ambient_noise may have read it
            # Re-open to read full audio reliably
        # Re-open inside a new context to record properly
        with sr.AudioFile(audio_file_path) as source:
            audio = self.recognizer.record(source)

        # First try without a strong language hint
        text = self.speech_to_text(audio, language_hint=None)

        # If empty, try other hints (Malayalam first)
        if not text:
            # try Malayalam
            text = self.speech_to_text(audio, language_hint="ml")
        if not text:
            # try Manglish (English recognizer)
            text = self.speech_to_text(audio, language_hint="manglish")

        if text:
            detected = self.language_detector.detect_language(text)
            return text, detected
        else:
            # No transcription available
            return "", "en"


# Example usage:
if __name__ == "__main__":
    vp = VoiceProcessor()
    # Replace 'sample.wav' with your audio file path
    test_audio = "sample.wav"
    try:
        txt, lang = vp.process_audio_file(test_audio)
        print("Transcribed:", txt)
        print("Detected language:", lang)

        if txt:
            mp3_path = vp.text_to_speech(f"Detected answer for testing: {txt}", language=lang, play_audio=False)
            print("Saved TTS to:", mp3_path)
    except FileNotFoundError:
        print(f"Put a test audio file at: {test_audio}")
    except Exception as e:
        print("Error:", e)
