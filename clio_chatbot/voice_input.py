"""Voice input for Clio using Whisper speech-to-text."""

import os
import sys
import numpy as np
import queue
import threading
from contextlib import contextmanager
from scipy.signal import resample
from typing import Optional, Callable


@contextmanager
def suppress_alsa_errors():
    """Suppress ALSA error messages that spam the console."""
    # Save the original stderr
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


# Import pyaudio with suppressed ALSA errors
with suppress_alsa_errors():
    import pyaudio

# Lazy imports for faster startup
WhisperModel = None
load_vad = None

# Audio settings
MIC_SAMPLE_RATE = 44100  # Native mic rate
WHISPER_SAMPLE_RATE = 16000  # What Whisper expects
CHUNK_DURATION = 0.096  # ~96ms chunks
CHUNK_SIZE = int(MIC_SAMPLE_RATE * CHUNK_DURATION)
WHISPER_CHUNK_SIZE = int(WHISPER_SAMPLE_RATE * CHUNK_DURATION)

# VAD settings
SPEECH_THRESHOLD = 0.5
SILENCE_CHUNKS = 52  # Chunks of silence before ending speech (~5 seconds)


def _load_models():
    """Lazy load the ML models."""
    global WhisperModel, load_vad
    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        from whisper_trt.vad import load_vad as lv
        WhisperModel = WM
        load_vad = lv


class VoiceInput:
    """Speech-to-text input using Whisper and VAD."""

    def __init__(self, model_size: str = "tiny.en", device_index: Optional[int] = None):
        self.model_size = model_size
        self.device_index = device_index
        self.model = None
        self.vad = None
        self.audio_queue = queue.Queue()
        self.running = False
        self._loaded = False
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None

    def load(self):
        """Load models (can be called ahead of time to warm up)."""
        if self._loaded:
            return

        _load_models()
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        self.vad = load_vad()
        self._loaded = True

    def find_usb_mic(self) -> Optional[int]:
        """Find the first USB audio input device."""
        with suppress_alsa_errors():
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    name = info['name'].lower()
                    if 'usb' in name or 'cmteck' in name:
                        p.terminate()
                        return i
            p.terminate()
            return None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - puts audio chunks in queue."""
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def listen_once(self, timeout: float = 120.0) -> Optional[str]:
        """
        Listen for a single utterance and return the transcribed text.
        Blocks until speech is detected and completed, or timeout.

        Returns None if no speech detected within timeout.
        """
        if not self._loaded:
            self.load()

        if self.device_index is None:
            self.device_index = self.find_usb_mic()

        if self.device_index is None:
            raise RuntimeError("No USB microphone found!")

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Open audio stream with suppressed ALSA errors
        with suppress_alsa_errors():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=MIC_SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )

        speech_buffer = []
        silence_count = 0
        is_speaking = False
        result_text = None

        self.running = True
        stream.start_stream()

        import time
        start_time = time.time()

        try:
            while self.running:
                # Timeout only applies while waiting for speech to START
                if not is_speaking and (time.time() - start_time) >= timeout:
                    break  # Timed out waiting for speech

                try:
                    audio_bytes = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Convert and resample
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_resampled = resample(audio_np, WHISPER_CHUNK_SIZE)

                # Run VAD
                voice_prob = float(self.vad(audio_resampled, sr=WHISPER_SAMPLE_RATE).flatten()[0])

                if voice_prob > SPEECH_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        if self._on_speech_start:
                            self._on_speech_start()
                    speech_buffer.append(audio_resampled)
                    silence_count = 0
                elif is_speaking:
                    speech_buffer.append(audio_resampled)
                    silence_count += 1

                    if silence_count >= SILENCE_CHUNKS:
                        # End of speech - transcribe
                        if self._on_speech_end:
                            self._on_speech_end()

                        audio_data = np.concatenate(speech_buffer)
                        segments, _ = self.model.transcribe(
                            audio_data,
                            beam_size=5,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            without_timestamps=True
                        )
                        # Get unique text (avoid duplicates from overlapping segments)
                        texts = []
                        for seg in segments:
                            text = seg.text.strip()
                            if text and (not texts or text != texts[-1]):
                                texts.append(text)
                        result_text = " ".join(texts)
                        break

        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()

        return result_text if result_text else None

    def on_speech_start(self, callback: Callable):
        """Set callback for when speech starts."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable):
        """Set callback for when speech ends."""
        self._on_speech_end = callback


# Singleton instance for reuse
_voice_input: Optional[VoiceInput] = None


def get_voice_input() -> VoiceInput:
    """Get or create the shared VoiceInput instance."""
    global _voice_input
    if _voice_input is None:
        _voice_input = VoiceInput()
    return _voice_input
