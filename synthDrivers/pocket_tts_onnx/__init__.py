import os
import sys
import threading
import queue
import ctypes
import glob
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict

# NVDA Core Imports
import globalVars
from nvwave import WavePlayer, AudioPurpose
from logHandler import log
import synthDriverHandler
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    NumericDriverSetting,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, VolumeCommand, BreakCommand

# --- PATH CONFIGURATION ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(os.path.dirname(DRIVER_DIR))
LIBS_DIR = os.path.join(ADDON_DIR, "libs")

if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)

try:
    import numpy as np
    from .pocket_tts_onnx import PocketTTSOnnx
    log.info("Pocket TTS ONNX: Numpy and engine loaded.")
except ImportError as e:
    log.error(f"Pocket TTS ONNX: Error loading dependencies: {e}")
    np = None
    PocketTTSOnnx = None

# =========================================================================
# SYNTHESIS QUEUE THREAD
# =========================================================================

class _SynthQueueThread(threading.Thread):
    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_event = threading.Event()

    def run(self):
        ctypes.windll.ole32.CoInitialize(None)
        while not self.stop_event.is_set():
            try:
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_request(request)
            except Exception as e:
                log.error(f"Pocket TTS ONNX: Synthesis error: {e}")
                try:
                    gen, segments, _voice_path = request
                    indices = [
                        index
                        for _text, segment_indices in segments
                        for index in segment_indices
                    ]
                    self._notify_finished(gen, indices)
                finally:
                    self.driver._request_queue.task_done()
            else:
                self.driver._request_queue.task_done()

        ctypes.windll.ole32.CoUninitialize()

    def _notify_finished(self, gen, indices):
        if gen != self.driver._speech_gen:
            return
        for idx in (indices or []):
            synthIndexReached.notify(synth=self.driver, index=idx)
        synthDoneSpeaking.notify(synth=self.driver)

    def _process_request(self, request):
        driver = self.driver
        gen, segments, voice_path = request
        if gen != driver._speech_gen:
            return

        self.cancel_event.clear()
        if gen != driver._speech_gen or self.cancel_event.is_set() or not driver.tts_engine:
            indices = [
                index
                for _text, segment_indices in segments
                for index in segment_indices
            ]
            self._notify_finished(gen, indices)
            return

        def fire_notifications(indices, done=False):
            if gen != driver._speech_gen:
                return
            for index in indices:
                synthIndexReached.notify(synth=driver, index=index)
            if done:
                synthDoneSpeaking.notify(synth=driver)

        volume_factor = driver._volume / 100.0
        last_segment = len(segments) - 1
        for segment_number, (text, indices) in enumerate(segments):
            is_last = segment_number == last_segment
            fed_audio = False
            if text and text.strip():
                audio_stream = driver.tts_engine.stream(
                    text=text,
                    voice=voice_path,
                    target_buffer_sec=0.2,
                    cancel_event=self.cancel_event,
                )

                for chunk in audio_stream:
                    if self.cancel_event.is_set() or gen != driver._speech_gen:
                        break
                    if chunk is not None and driver._player:
                        pcm = np.asarray(np.clip(chunk * volume_factor, -1.0, 1.0))
                        if pcm.size == 0:
                            continue
                        driver._player.feed(
                            (pcm * 32767).astype(np.int16).tobytes(),
                        )
                        fed_audio = True

            if self.cancel_event.is_set() or gen != driver._speech_gen:
                return

            if fed_audio and (indices or is_last):
                driver._player.idle()
            if self.cancel_event.is_set() or gen != driver._speech_gen:
                return
            fire_notifications(indices, is_last)


# =========================================================================
# MAIN SYNTHDRIVER CLASS
# =========================================================================

class SynthDriver(BaseSynthDriver):
    name = "pocket_tts_onnx"
    description = "Pocket TTS ONNX Synthesizer"

    @classmethod
    def check(cls):
        return np is not None and PocketTTSOnnx is not None

    def __init__(self):
        super(SynthDriver, self).__init__()

        if globalVars.appArgs.secure:
            self.models_root = os.path.join(r"C:\Program Files\NVDA\systemConfig", "pocket_tts")
        else:
            self.models_root = os.path.join(globalVars.appArgs.configPath, "pocket_tts")

        self.models_dir = os.path.join(self.models_root, "onnx")
        self.tokenizer_path = os.path.join(self.models_root, "tokenizer.model")
        self.voices_dir = os.path.join(self.models_root, "voices")

        if not globalVars.appArgs.secure:
            os.makedirs(self.voices_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)

        self._current_voice_id = ""
        self._current_voice_path = None  # str path passed directly to the engine
        self._volume = 80
        self._eos_threshold = -2.0
        self._lsd_steps = 10
        self.tts_engine = None
        self._player = None
        self._available_voices = OrderedDict()
        self._request_queue = queue.Queue()
        self._engine_loaded_event = threading.Event()
        self._speech_gen = 0

        self._scan_voices()
        self._worker_thread = _SynthQueueThread(driver=self)
        self._worker_thread.start()
        threading.Thread(target=self._initialize_async, daemon=True).start()

    supportedCommands = frozenset([IndexCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])
    supportedSettings = (
        BaseSynthDriver.VoiceSetting(),
        BaseSynthDriver.VolumeSetting(),
        NumericDriverSetting(
            "eosThreshold",
            # Translators: Label for the EOS sensitivity setting in NVDA speech settings
            _("End-of-sentence sensitivity (EOS)"),
            availableInSettingsRing=True,
            minVal=0,
            maxVal=100,
        ),
        NumericDriverSetting(
            "lsdSteps",
            # Translators: Label for the Pocket TTS flow matching step setting in NVDA speech settings
            _("Flow steps"),
            availableInSettingsRing=True,
            defaultVal=10,
            minVal=1,
            maxVal=10,
            normalStep=1,
            largeStep=2,
        ),
    )

    def _scan_voices(self):
        """Scan for .wav and .npy voice files in the user voices folder."""
        self._available_voices.clear()
        files = (
            glob.glob(os.path.join(self.voices_dir, "*.npy"))
            + glob.glob(os.path.join(self.voices_dir, "*.wav"))
        )
        seen = set()
        for path in files:
            name, ext = os.path.splitext(os.path.basename(path))
            if name not in seen:
                self._available_voices[name] = VoiceInfo(
                    name, name.replace("_", " ").title(), "en"
                )
                seen.add(name)

        if self._available_voices and not self._current_voice_id:
            first = list(self._available_voices.keys())[0]
            self._current_voice_id = first
            self._current_voice_path = self._resolve_voice_path(first)

    def _resolve_voice_path(self, voice_id: str) -> str:
        """Return the full path for a voice id, preferring .npy over .wav."""
        for ext in (".npy", ".wav"):
            path = os.path.join(self.voices_dir, f"{voice_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    def _initialize_async(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            self._player = WavePlayer(
                channels=1,
                samplesPerSec=24000,
                bitsPerSample=16,
                purpose=AudioPurpose.SPEECH,
            )
            # Start with the model's full-quality flow setting.
            self.tts_engine = PocketTTSOnnx(
                models_dir=self.models_dir,
                tokenizer_path=self.tokenizer_path,
                precision="int8",
                lsd_steps=self._lsd_steps,
                eos_threshold=self._eos_threshold,
            )
            self._engine_loaded_event.set()
            log.info("Pocket TTS ONNX: Ready for use.")
        except Exception as e:
            log.error(f"Pocket TTS ONNX: Initialization failed: {e}")

    # --- Voice property ---

    def _get_availableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        return self._available_voices

    def _get_voice(self):
        return self._current_voice_id

    def _set_voice(self, value):
        if value in self._available_voices:
            self._current_voice_id = value
            self._current_voice_path = self._resolve_voice_path(value)

    # --- Volume property ---

    def _get_volume(self):
        return self._volume

    def _set_volume(self, value):
        self._volume = value

    # --- EOS threshold property ---
    # Exposed as a 0-100 slider in NVDA speech settings.
    # Internally mapped to the logit range [-4.0, 0.0]:
    #   slider 0   = logit -4.0  (most sensitive, may stop early)
    #   slider 50  = logit -2.0  (recommended default)
    #   slider 100 = logit  0.0  (least sensitive, always finishes sentence)

    def _get_eosThreshold(self):
        # Map internal logit [-4.0, 0.0] -> slider [0, 100]
        return int((self._eos_threshold + 4.0) / 4.0 * 100)

    def _set_eosThreshold(self, value):
        # Map slider [0, 100] -> logit [-4.0, 0.0]
        self._eos_threshold = (value / 100.0) * 4.0 - 4.0
        if self.tts_engine is not None:
            self.tts_engine.eos_threshold = self._eos_threshold

    # --- Flow steps property ---

    def _get_lsdSteps(self):
        return self._lsd_steps

    def _set_lsdSteps(self, value):
        self._lsd_steps = max(1, min(10, int(value)))
        if self.tts_engine is not None:
            self.tts_engine.set_lsd_steps(self._lsd_steps)

    # --- Speech ---

    def speak(self, speechSequence):
        """Queue text in index-aligned segments for spelling and Say All."""
        if not self._engine_loaded_event.is_set() or self._current_voice_path is None:
            return

        segments = []
        text_parts = []

        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, IndexCommand):
                text = "".join(text_parts)
                if text or not segments:
                    segments.append((text, [item.index]))
                else:
                    segments[-1][1].append(item.index)
                text_parts.clear()
            elif isinstance(item, VolumeCommand):
                self._volume = item.value

        trailing_text = "".join(text_parts)
        if trailing_text.strip() or not segments:
            segments.append((trailing_text, []))
        if any(text.strip() or indices for text, indices in segments):
            self._request_queue.put((self._speech_gen, segments, self._current_voice_path))

    def pause(self, switch):
        """Pause or resume playback. Called by NVDA when the user presses Shift."""
        if self._player:
            try:
                self._player.pause(switch)
            except Exception:
                pass

    def cancel(self):
        self._speech_gen += 1
        self._worker_thread.cancel_event.set()
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
                self._request_queue.task_done()
            except queue.Empty:
                break

    def terminate(self):
        self._speech_gen += 1
        self._worker_thread.cancel_event.set()
        self._worker_thread.stop_event.set()
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        # Join the worker thread before releasing resources so it cannot
        # access _player or tts_engine after they have been freed.
        self._worker_thread.join(timeout=2.0)
        if self._player:
            self._player.close()
        self.tts_engine = None
