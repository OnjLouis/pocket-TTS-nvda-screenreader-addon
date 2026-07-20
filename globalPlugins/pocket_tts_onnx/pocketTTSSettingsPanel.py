import os
import sys
import threading
import time
import wx
import gui
import globalVars
import ui
from logHandler import log
from gui.settingsDialogs import SettingsPanel

# Locate the ONNX engine relative to this file:
# globalPlugins/pocket_tts_onnx/ -> addon root -> synthDrivers/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SYNTH_DIR = os.path.join(ADDON_DIR, "synthDrivers")
LIBS_DIR = os.path.join(ADDON_DIR, "libs")

for _dir in (SYNTH_DIR, LIBS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

try:
    import numpy as np
    from pocket_tts_onnx import PocketTTSOnnx
    log.info("Pocket TTS Settings: engine loaded.")
except ImportError as e:
    log.error(f"Pocket TTS Settings: Error loading dependencies: {e}")
    np = None
    PocketTTSOnnx = None

_ = lambda s: s


class PocketTTSSettingsPanel(SettingsPanel):
    title = _("Pocket TTS Voice Manager")

    def makeSettings(self, settingsSizer):
        conf_dir = globalVars.appArgs.configPath
        self.models_root = os.path.join(conf_dir, "pocket_tts")
        self.voices_dir = os.path.join(self.models_root, "voices")
        self.onnx_dir = os.path.join(self.models_root, "onnx")
        self.tokenizer_path = os.path.join(self.models_root, "tokenizer.model")

        os.makedirs(self.voices_dir, exist_ok=True)
        self._voice_search_prefix = ""
        self._voice_search_time = 0.0

        # --- Section 1: Add / Clone Voice ---
        add_box = wx.StaticBox(self, label=_("Add new voice (Voice Cloning)"))
        add_sizer = wx.StaticBoxSizer(add_box, wx.VERTICAL)

        help_text = wx.StaticText(
            self,
            label=_(
                "Select an audio file (MP3/WAV) to generate a new .npy voice embedding.\n"
                "Only the first 30 seconds of the file will be used."
            ),
        )
        add_sizer.Add(help_text, 0, wx.ALL, 5)

        self.btnAdd = wx.Button(self, label=_("&Convert audio file..."))
        self.btnAdd.Bind(wx.EVT_BUTTON, self.onAddVoice)
        add_sizer.Add(self.btnAdd, 0, wx.ALL, 5)
        settingsSizer.Add(add_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # --- Section 2: Manage Voices ---
        manage_box = wx.StaticBox(self, label=_("Manage installed voices"))
        manage_sizer = wx.StaticBoxSizer(manage_box, wx.VERTICAL)

        list_label = wx.StaticText(self, label=_("&Select voice:"))
        manage_sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 5)

        self.voiceList = wx.Choice(self, choices=self._get_installed_voices())
        self.voiceList.Bind(wx.EVT_CHOICE, self.onVoiceSelect)
        self.voiceList.Bind(wx.EVT_CHAR_HOOK, self.onVoiceListChar)
        manage_sizer.Add(self.voiceList, 0, wx.EXPAND | wx.ALL, 5)

        name_label = wx.StaticText(self, label=_("&Display name:"))
        manage_sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 5)

        self.nameEdit = wx.TextCtrl(self)
        manage_sizer.Add(self.nameEdit, 0, wx.EXPAND | wx.ALL, 5)

        self.btnRename = wx.Button(self, label=_("Re&name voice file"))
        self.btnRename.Bind(wx.EVT_BUTTON, self.onRenameVoice)
        manage_sizer.Add(self.btnRename, 0, wx.ALL, 5)

        manage_sizer.AddSpacer(10)

        self.btnRemove = wx.Button(self, label=_("Remo&ve selected voice"))
        self.btnRemove.Bind(wx.EVT_BUTTON, self.onRemoveVoice)
        manage_sizer.Add(self.btnRemove, 0, wx.ALL, 5)

        settingsSizer.Add(manage_sizer, 0, wx.EXPAND | wx.ALL, 10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_installed_voices(self):
        """Return list of voice files in the voices directory."""
        if not os.path.exists(self.voices_dir):
            return []
        try:
            return [
                f for f in os.listdir(self.voices_dir)
                if f.lower().endswith((".npy", ".wav"))
            ]
        except Exception:
            return []

    def _refresh_ui(self, selected_voice=None):
        """Reload the voice list and optionally select a specific voice."""
        voices = self._get_installed_voices()
        self.voiceList.Clear()
        self.voiceList.AppendItems(voices)
        if voices:
            selection = 0
            if selected_voice in voices:
                selection = voices.index(selected_voice)
            self.voiceList.SetSelection(selection)
            self.onVoiceSelect(None)
        else:
            self.nameEdit.Clear()

    def _select_voice_by_prefix(self, prefix, announce=False):
        """Select the first voice matching prefix; returns True if handled."""
        prefix = prefix.lower()
        for idx in range(self.voiceList.GetCount()):
            voice = self.voiceList.GetString(idx)
            base = os.path.splitext(voice)[0].lower()
            if voice.lower().startswith(prefix) or base.startswith(prefix):
                self.voiceList.SetSelection(idx)
                self.onVoiceSelect(None)
                if announce:
                    ui.message(os.path.splitext(voice)[0])
                return True
        return False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def onAddVoice(self, evt):
        """Encode an audio file and save the resulting .npy embedding."""
        if PocketTTSOnnx is None or np is None:
            gui.messageBox(
                _("Dependencies not loaded. Check the NVDA log for details."),
                _("Error"),
            )
            return

        wildcard = "Audio files (*.mp3;*.wav)|*.mp3;*.wav"
        with wx.FileDialog(
            self,
            message=_("Select a voice sample"),
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fd:
            if fd.ShowModal() != wx.ID_OK:
                return
            audio_src = fd.GetPath()

        base_name = os.path.splitext(os.path.basename(audio_src))[0]
        dest_path = os.path.join(self.voices_dir, f"{base_name}.npy")

        if os.path.exists(dest_path):
            gui.messageBox(
                _("A voice named '{name}' already exists. Rename the source file or remove the existing voice first.").format(name=base_name),
                _("Voice already exists"),
            )
            return

        self.btnAdd.Disable()
        self.btnAdd.SetLabel(_("Converting voice..."))

        def worker():
            try:
                tts = PocketTTSOnnx(
                    models_dir=self.onnx_dir,
                    tokenizer_path=self.tokenizer_path,
                    precision="int8",
                    lsd_steps=1,
                )
                embedding = tts.encode_voice(audio_src)
                np.save(dest_path, embedding)
            except Exception as e:
                log.error(f"Pocket TTS: Conversion error: {e}")
                wx.CallAfter(self._finish_voice_conversion, base_name, e)
            else:
                wx.CallAfter(self._finish_voice_conversion, base_name, None)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_voice_conversion(self, base_name, error):
        self.btnAdd.Enable()
        self.btnAdd.SetLabel(_("&Convert audio file..."))
        if error is not None:
            gui.messageBox(f"Error during conversion:\n{error}", _("Error"))
            return
        selected_voice = f"{base_name}.npy"
        self._refresh_ui(selected_voice=selected_voice)
        gui.messageBox(
            _("Voice '{name}' successfully created!").format(name=base_name),
            _("Success"),
        )
        wx.CallAfter(self.nameEdit.SetFocus)
        wx.CallAfter(self.nameEdit.SelectAll)

    def onRenameVoice(self, evt):
        """Rename the selected .npy file on disk."""
        old_filename = self.voiceList.GetStringSelection()
        new_name = self.nameEdit.GetValue().strip()
        if not old_filename or not new_name:
            return

        old_path = os.path.join(self.voices_dir, old_filename)
        _old_base, old_ext = os.path.splitext(old_filename)
        new_path = os.path.join(self.voices_dir, f"{new_name}.npy")
        if old_ext.lower() == ".wav":
            new_path = os.path.join(self.voices_dir, f"{new_name}.wav")

        if os.path.exists(new_path):
            gui.messageBox(
                _("A voice named '{name}' already exists.").format(name=os.path.basename(new_path)),
                _("Voice already exists"),
            )
            return

        try:
            os.rename(old_path, new_path)
            self._refresh_ui(selected_voice=os.path.basename(new_path))
            gui.messageBox(
                _("Voice renamed to '{name}'.").format(name=new_name),
                _("Success"),
            )
        except Exception as err:
            gui.messageBox(f"Rename failed.\n{err}", _("Error"))

    def onRemoveVoice(self, evt):
        """Delete the selected .npy file after confirmation."""
        sel = self.voiceList.GetStringSelection()
        if not sel:
            return

        if (
            gui.messageBox(
                _("Are you sure you want to remove the voice '{name}'?").format(name=sel),
                _("Confirm"),
                wx.YES_NO | wx.ICON_QUESTION,
            )
            == wx.YES
        ):
            try:
                os.remove(os.path.join(self.voices_dir, sel))
                self._refresh_ui()
            except Exception as e:
                log.error(f"Pocket TTS: Error removing voice: {e}")

    def onVoiceSelect(self, evt):
        """Populate the name field when a voice is selected."""
        filename = self.voiceList.GetStringSelection()
        if filename:
            self.nameEdit.SetValue(os.path.splitext(filename)[0])

    def onVoiceListChar(self, evt):
        """Allow multi-letter prefix navigation in the voice list."""
        if evt.ControlDown() or evt.AltDown() or evt.MetaDown():
            evt.Skip()
            return

        key = evt.GetKeyCode()
        if key in (wx.WXK_BACK, wx.WXK_DELETE, wx.WXK_ESCAPE):
            self._voice_search_prefix = ""
            self._voice_search_time = 0.0
            evt.Skip()
            return

        char = evt.GetUnicodeKey()
        if char == wx.WXK_NONE:
            evt.Skip()
            return
        typed = chr(char)
        if not typed.isprintable() or typed.isspace():
            evt.Skip()
            return

        now = time.monotonic()
        if now - self._voice_search_time > 1.2:
            self._voice_search_prefix = ""
        self._voice_search_time = now

        candidate = self._voice_search_prefix + typed.lower()
        if self._select_voice_by_prefix(candidate, announce=True):
            self._voice_search_prefix = candidate
            return
        if self._select_voice_by_prefix(typed, announce=True):
            self._voice_search_prefix = typed.lower()
            return

        wx.Bell()

    def onSave(self):
        pass
