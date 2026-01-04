"""Voice output for Clio with smart TTS mode."""

import re
import subprocess
from pathlib import Path


SPEAK_SCRIPT = Path("/tmp/speak.sh")
WORD_THRESHOLD = 100  # Speak full if under this many words


class Voice:
    """Smart voice output that adapts to content length."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._check_available()

    def _check_available(self) -> bool:
        """Check if TTS is available."""
        if not SPEAK_SCRIPT.exists():
            self.enabled = False
            return False
        return True

    def speak(self, text: str, force_full: bool = False):
        """Speak text using smart mode."""
        if not self.enabled:
            return

        # Clean the text for speech
        speech_text = self._prepare_for_speech(text)

        if not speech_text.strip():
            return

        # Decide what to speak based on length
        word_count = len(speech_text.split())

        if force_full or word_count <= WORD_THRESHOLD:
            self._speak_text(speech_text)
        else:
            # Summarize for long responses
            summary = self._create_speech_summary(text, word_count)
            self._speak_text(summary)

    def _prepare_for_speech(self, text: str) -> str:
        """Prepare text for TTS by removing code blocks, URLs, etc."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '[code block]', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
        text = re.sub(r'#+\s*', '', text)                # Headers

        # Remove bullet points but keep content
        text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _create_speech_summary(self, text: str, word_count: int) -> str:
        """Create a TTS-friendly summary of long content."""
        # Get the first meaningful sentence
        sentences = re.split(r'[.!?]+', text)
        first_sentence = sentences[0].strip() if sentences else ""

        # Count items if it's a list
        list_items = re.findall(r'^\s*[-*]\s+', text, re.MULTILINE)
        item_count = len(list_items)

        if item_count > 0:
            return f"{first_sentence}. I have {item_count} points to share. Would you like me to go through them?"
        elif word_count > 200:
            return f"{first_sentence}. I have more details if you'd like to hear them."
        else:
            # Just take first ~100 words
            words = text.split()[:100]
            return " ".join(words) + "..."

    def _speak_text(self, text: str):
        """Actually invoke the TTS script."""
        if not text.strip():
            return

        try:
            # Escape single quotes for shell
            escaped = text.replace("'", "'\\''")
            subprocess.Popen(
                [str(SPEAK_SCRIPT), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            pass  # Silently fail if TTS unavailable

    def speak_greeting(self, time_since: str = None, is_first: bool = False):
        """Speak a contextual greeting."""
        if is_first:
            self.speak("Hello! I'm Clio. It's nice to meet you.")
        elif time_since:
            self.speak(f"Welcome back! It's been {time_since} since we last talked.")
        else:
            self.speak("Hey! Good to see you again.")

    def speak_farewell(self):
        """Speak a farewell message."""
        self.speak("Take care! I'll remember our conversation.")
