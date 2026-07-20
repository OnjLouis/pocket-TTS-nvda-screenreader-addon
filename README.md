## NVDA screenreader Pocket TTS add-on
This is a personal maintained fork of the Pocket TTS NVDA add-on for NVDA 2026.1 and later. It uses [Kyutai Labs Pocket TTS](https://github.com/kyutai-labs/pocket-tts), based on the ONNX int8 versions of [Pocket TTS by KevinAHM](https://huggingface.co/KevinAHM/pocket-tts-onnx/tree/main/onnx).

* English language support using the v1 Pocket TTS model.
* Voice cloning from WAV or MP3 samples in the add-on settings.
* Bundled voices copied into the user's NVDA configuration on install.
* Memory-conscious int8 ONNX model support.

## Changes in 1.1.3

* Restored reliable spelling and Say All navigation by synchronizing NVDA indexes with completed playback.
* Prevented empty model input from stopping speech on some characters and symbols.
* Corrected Flow steps to use the full 1 to 10 range, persist changes, and default to 10.

## Changes in 1.1.2

* Improved responsiveness by caching voice-conditioning state and overlapping speech generation with audio decoding.
* Reduced stale speech after interruption by tracking cancelled speech requests.
* Improved single-character punctuation handling.
* Added Flow steps to NVDA's speech settings.
* Improved the voice manager with non-blocking conversion, unique mnemonics, multi-letter voice-list search, and automatic selection of newly-created voices.
* Added GitHub release update checking for this maintained fork.
