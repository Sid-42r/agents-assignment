# src/interrupt_handler.py
"""
InterruptHandler (Python)
Simple, copy-paste class to decide whether a VAD event while agent is speaking
should be ignored or should interrupt the agent.

Usage: instantiate and register callbacks:
  ih = InterruptHandler(config)
  ih.on_validated_interrupt = lambda transcript: ...
  ih.on_ignore = lambda transcript, reason: ...
  ih.on_respond = lambda transcript: ...

Methods to call from your agent event handlers:
  ih.set_speaking(bool)
  ih.handle_vad_start()
  ih.handle_partial_transcript(text)
  ih.handle_final_transcript(text)

This file is a pure-Python logic layer; it does not depend on external libs.
"""

import threading
import time
from typing import Callable, List, Optional


class InterruptConfig:
    def __init__(
        self,
        ignore_list: Optional[List[str]] = None,
        interrupt_list: Optional[List[str]] = None,
        validation_timeout_ms: int = 150,
    ):
        self.ignore_list = ignore_list or ["yeah", "ok", "hmm", "uh-huh", "right", "mm", "mhm"]
        self.interrupt_list = interrupt_list or ["stop", "wait", "no", "pause", "hold", "hold on", "stop it"]
        self.validation_timeout_ms = validation_timeout_ms


class InterruptHandler:
    """
    Simple, synchronous-looking API (uses Timer internally).
    Events/callbacks you can set:
      - on_validated_interrupt(transcript)
      - on_ignore(transcript, reason)
      - on_respond(transcript)

    The agent should call:
      - set_speaking(True/False) when TTS starts/ends
      - handle_vad_start() when VAD detects voice
      - handle_partial_transcript(text) for STT partials
      - handle_final_transcript(text) for STT finals
    """

    def __init__(self, cfg: InterruptConfig):
        self.cfg = cfg
        self.is_speaking = False

        # Timer used to wait for STT. If it fires, we consider it a timeout => ignore.
        self._timer: Optional[threading.Timer] = None
        self._waiting_for_stt = False
        self._last_partial: str = ""

        # Callbacks (assign functions)
        self.on_validated_interrupt: Callable[[str], None] = lambda transcript: None
        self.on_ignore: Callable[[str, str], None] = lambda transcript, reason: None
        self.on_respond: Callable[[str], None] = lambda transcript: None

        # Lock to keep thread-safe
        self._lock = threading.Lock()

    # ---------- Public API ----------

    def set_speaking(self, speaking: bool):
        """Call when agent TTS starts/stops."""
        with self._lock:
            self.is_speaking = speaking
            # If we become silent while waiting for STT, flush as respond
            if not speaking and self._waiting_for_stt:
                # Treat as normal user input (agent silent)
                self._clear_timer_locked()
                partial = self._last_partial
                self._waiting_for_stt = False
                # respond with whatever partial we had (might be empty)
                self.on_respond(partial)

    def handle_vad_start(self):
        """Call immediately when VAD indicates user started speaking."""
        with self._lock:
            if not self.is_speaking:
                # Agent silent: normal conversation flow
                self.on_respond("")  # upstream will send STT and handle it
                return

            # Agent speaking: start short validation window (unless already waiting)
            if self._waiting_for_stt:
                return
            self._waiting_for_stt = True
            self._last_partial = ""
            timeout = self.cfg.validation_timeout_ms / 1000.0
            self._timer = threading.Timer(timeout, self._on_validation_timeout)
            self._timer.start()

    def handle_partial_transcript(self, partial_text: str):
        """Call with streaming partial transcripts."""
        text = (partial_text or "").strip().lower()
        with self._lock:
            self._last_partial = text

            if not self.is_speaking:
                # Agent silent -> immediate respond
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_respond(text)
                return

            if not self._waiting_for_stt:
                # Not waiting (maybe VAD didn't fire) -> ignore
                return

            # If partial contains explicit interrupt -> stop immediately
            if self._contains_interrupt(text):
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_validated_interrupt(text)
                return

            # If partial is clearly only filler
            if self._is_only_ignore_words(text):
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_ignore(text, "only-ignore-words")
                return

            # Otherwise: wait for more partial or final (do nothing now)

    def handle_final_transcript(self, final_text: str):
        """Call with final transcript."""
        text = (final_text or "").strip().lower()
        with self._lock:
            if not self.is_speaking:
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_respond(text)
                return

            if not self._waiting_for_stt:
                # No validation in progress; ignore
                return

            # If contains interrupt -> validated interrupt
            if self._contains_interrupt(text):
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_validated_interrupt(text)
                return

            # If only ignore words -> ignore
            if self._is_only_ignore_words(text):
                self._clear_timer_locked()
                self._waiting_for_stt = False
                self.on_ignore(text, "final-only-ignore")
                return

            # Mixed or contentful -> treat as interrupt
            self._clear_timer_locked()
            self._waiting_for_stt = False
            self.on_validated_interrupt(text)

    # ---------- Internal helpers ----------

    def _on_validation_timeout(self):
        """Called when validation timer expires (no STT arrived)."""
        with self._lock:
            self._timer = None
            if not self._waiting_for_stt:
                return
            self._waiting_for_stt = False
            # No STT arrived → ignore (assume filler)
            self.on_ignore(self._last_partial, "stt-timeout")

    def _clear_timer_locked(self):
        if self._timer:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None

    def _tokenize(self, s: str) -> List[str]:
        # simple tokenizer
        import re

        if not s:
            return []
        s = re.sub(r"[^\w\s'-]", " ", s)
        tokens = [t.strip() for t in s.split() if t.strip()]
        return [t.lower() for t in tokens]

    def _is_only_ignore_words(self, text: str) -> bool:
        tokens = self._tokenize(text)
        if not tokens:
            # empty partial -> can't be sure; return False to wait
            return False
        return all(tok in self.cfg.ignore_list for tok in tokens)

    def _contains_interrupt(self, text: str) -> bool:
        tokens = self._tokenize(text)
        # any explicit interrupt word
        for tok in tokens:
            if tok in self.cfg.interrupt_list:
                return True
        # Mixed input: any token that is not an ignore word => treat as interrupt
        for tok in tokens:
            if tok not in self.cfg.ignore_list:
                return True
        return False
