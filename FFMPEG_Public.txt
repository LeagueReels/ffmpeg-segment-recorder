#!/usr/bin/env python3
# Copyright (c) 2026 LeagueReels — MIT License (see LICENSE)
"""
FFmpeg Segment Recorder
=======================

Rolling-buffer desktop recorder using FFmpeg *gdigrab* and the *segment*
muxer.  Keeps a configurable window of ``.ts`` files on disk so that a
clip can be extracted from the buffer at any time.

Usage (see --help for all flags):

    # Basic recording (30 fps, 130-second buffer):
    python ffmpeg_segment_recorder.py

    # Custom output folder, 60 fps, 10-minute buffer:
    python ffmpeg_segment_recorder.py --out-dir C:/captures --fps 60 --buffer-seconds 600

    # List available DirectShow audio devices, then exit:
    python ffmpeg_segment_recorder.py --list-audio-devices

    # Record with audio:
    python ffmpeg_segment_recorder.py --audio --audio-device "Stereo Mix (Realtek Audio)"

    # Dry-run (print FFmpeg command, do not execute):
    python ffmpeg_segment_recorder.py --dry-run

Requirements:
    pip install imageio-ffmpeg

Python: 3.9+
Platform: Windows (gdigrab).  get_physical_screen_resolution() falls back
          gracefully on Linux/macOS.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# =============================================================================
# WINDOWS SUBPROCESS HELPERS
# =============================================================================

def get_subprocess_windows_flags():
    """
    Return Windows flags that suppress a console window for FFmpeg
    subprocesses.

    Returns:
        ``(creation_flags, startupinfo)`` — both are ``0``/``None`` on
        non-Windows so callers can pass them unconditionally.
    """
    if sys.platform != "win32":
        return 0, None

    creation_flags = subprocess.CREATE_NO_WINDOW
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return creation_flags, si


# =============================================================================
# PHYSICAL SCREEN RESOLUTION
# =============================================================================

def get_physical_screen_resolution():
    """
    Return the physical screen resolution in pixels, bypassing Windows DPI
    scaling.

    On high-DPI displays Windows can report a *scaled* resolution while
    FFmpeg's ``gdigrab`` captures at the true hardware resolution.  Passing
    the wrong ``-video_size`` produces a cropped or corrupted recording.

    Returns:
        ``(width, height)`` integers.  Falls back to ``(1920, 1080)`` if
        the query fails or the platform is not Windows.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception as exc:
            logging.warning(f"Could not detect physical screen resolution: {exc}")

    return 1920, 1080  # safe fallback


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    Parameters consumed by :class:`SegmentRecorder` and
    :class:`RingBufferPruner`.

    Example::

        config = Config(output_base=Path("output"))
        config.ffmpeg_exe = find_ffmpeg(config)
        config.create_directories()
    """

    # Paths
    output_base: Path = field(default_factory=lambda: Path("output"))
    buffer_dir: Path = field(init=False)

    # FFmpeg executable (resolved by find_ffmpeg)
    ffmpeg_exe: Optional[str] = None

    # Capture
    capture_fps: int = 30
    segment_duration: int = 8       # seconds per .ts file

    # Ring buffer
    buffer_seconds: int = 130
    safety_seconds: int = 30        # extra margin on top of buffer

    # Audio (DirectShow / Stereo Mix — Windows only)
    include_audio: bool = False
    audio_device_name: Optional[str] = None
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    audio_bitrate: str = "192k"

    # Misc
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.buffer_dir = self.output_base / "buffer"

    def create_directories(self) -> None:
        """Create the buffer directory (and any parents)."""
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

    @property
    def total_buffer_seconds(self) -> int:
        """Nominal buffer + safety margin, in seconds."""
        return self.buffer_seconds + self.safety_seconds


# =============================================================================
# FIND FFMPEG
# =============================================================================

def find_ffmpeg(config: Config) -> str:
    """
    Locate the FFmpeg executable (priority order):

    1. ``imageio_ffmpeg.get_ffmpeg_exe()`` — preferred bundled binary.
    2. ``config.ffmpeg_exe`` — manually configured path.
    3. ``RuntimeError`` if neither works.

    Args:
        config: Only ``config.ffmpeg_exe`` is read as a fallback.

    Returns:
        Absolute path string to a working ``ffmpeg`` binary.
    """
    creation_flags, si = get_subprocess_windows_flags()

    def _verify(path: str) -> bool:
        try:
            r = subprocess.run(
                [path, "-version"],
                capture_output=True,
                timeout=5,
                creationflags=creation_flags,
                startupinfo=si,
            )
            return r.returncode == 0
        except Exception:
            return False

    # 1. imageio_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import]
        fp = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(fp) and os.path.getsize(fp) > 0 and _verify(fp):
            logging.info(f"FFmpeg found via imageio_ffmpeg: {fp}")
            return fp
        logging.warning(f"imageio_ffmpeg returned an unusable path: {fp}")
    except ImportError:
        logging.warning("imageio_ffmpeg not installed — run: pip install imageio-ffmpeg")
    except Exception as exc:
        logging.warning(f"imageio_ffmpeg error: {exc}")

    # 2. Manual fallback
    if config.ffmpeg_exe and os.path.exists(config.ffmpeg_exe):
        if _verify(config.ffmpeg_exe):
            logging.info(f"Using manually configured FFmpeg: {config.ffmpeg_exe}")
            return config.ffmpeg_exe
        logging.error(f"Manually configured FFmpeg not executable: {config.ffmpeg_exe}")

    raise RuntimeError(
        "FFmpeg not found.\n"
        "  Option A: pip install imageio-ffmpeg\n"
        "  Option B: pass --ffmpeg /path/to/ffmpeg.exe\n"
    )


# =============================================================================
# LIST DIRECTSHOW AUDIO DEVICES
# =============================================================================

def list_audio_devices(ffmpeg_exe: str) -> None:
    """
    Run ``ffmpeg -list_devices true -f dshow -i dummy`` and print the output.

    This is the standard way to enumerate available DirectShow audio (and
    video) devices on Windows.  The command always exits with a non-zero
    code; that is expected — the device list is written to stderr before
    FFmpeg tries (and fails) to open ``dummy``.
    """
    creation_flags, si = get_subprocess_windows_flags()
    cmd = [ffmpeg_exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]

    print("Querying DirectShow devices — this is a local read-only enumeration.\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=creation_flags,
            startupinfo=si,
        )
        # FFmpeg prints device list to stderr
        output = result.stderr or result.stdout
        print(output)
    except Exception as exc:
        print(f"ERROR: Could not run FFmpeg: {exc}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# RUN FFMPEG  (thin Popen wrapper)
# =============================================================================

def run_ffmpeg(
    cmd: List[str],
    dry_run: bool = False,
    log_output: bool = False,
    stdin_pipe: bool = False,
) -> Optional[subprocess.Popen]:
    """
    Start an FFmpeg process, hiding its console window on Windows.

    Args:
        cmd:        Full command list (executable first).
        dry_run:    Log the command and return ``None`` without executing.
        log_output: Attach ``PIPE`` to stdout/stderr for the caller to read.
        stdin_pipe: Attach ``PIPE`` to stdin (e.g. to send ``q`` to stop).

    Returns:
        ``subprocess.Popen`` or ``None`` when ``dry_run=True`` / on error.
    """
    if dry_run:
        logging.info("[DRY-RUN] %s", " ".join(cmd))
        return None

    creation_flags, si = get_subprocess_windows_flags()
    stdin = subprocess.PIPE if stdin_pipe else None

    try:
        if log_output:
            return subprocess.Popen(
                cmd,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creation_flags,
                startupinfo=si,
            )
        return subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            startupinfo=si,
        )
    except Exception as exc:
        logging.error(f"Failed to start FFmpeg: {exc}")
        return None


# =============================================================================
# SEGMENT RECORDER
# =============================================================================

class SegmentRecorder:
    """
    Captures the Windows desktop to a rolling ring of ``.ts`` segment files
    using FFmpeg's ``gdigrab`` input device and the ``segment`` muxer.

    Each file is ``config.segment_duration`` seconds long and is stamped
    with the wall-clock time in its filename so the pruner can expire
    segments by modification time.

    Example::

        config = Config(output_base=Path("output"))
        config.ffmpeg_exe = find_ffmpeg(config)
        config.create_directories()

        recorder = SegmentRecorder(config)
        recorder.start()
        ...
        recorder.stop()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self.is_recording = False
        self.start_time: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin desktop capture.  No-op if already recording."""
        with self._lock:
            if self.is_recording:
                logging.warning("SegmentRecorder: already running")
                return

            screen_w, screen_h = get_physical_screen_resolution()
            logging.info(f"Screen resolution: {screen_w}x{screen_h}")

            cmd = self._build_command(screen_w, screen_h)

            logging.info("Starting desktop capture…")
            self.process = run_ffmpeg(
                cmd, dry_run=self.config.dry_run, stdin_pipe=True
            )

            if not self.config.dry_run and self.process:
                time.sleep(0.5)
                if self.process.poll() is not None:
                    logging.error(
                        "FFmpeg exited immediately — check resolution, "
                        "audio device name, or codec support"
                    )
                    self.is_recording = False
                    return

            self.is_recording = True
            self.start_time = time.time()
            logging.info(
                "Capture started  fps=%d  segment=%ds  audio=%s",
                self.config.capture_fps,
                self.config.segment_duration,
                self.config.include_audio,
            )

    def stop(self) -> None:
        """Stop desktop capture gracefully, falling back to kill if needed."""
        with self._lock:
            if not self.is_recording:
                return

            logging.info("Stopping desktop capture…")
            self.is_recording = False

            if self.process and not self.config.dry_run:
                self._stop_process()

            self.process = None

    def get_elapsed_seconds(self) -> float:
        """Seconds since ``start()`` was called, or ``0.0`` if not running."""
        return time.time() - self.start_time if self.start_time else 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_command(self, screen_w: int, screen_h: int) -> List[str]:
        cfg = self.config
        cmd = [cfg.ffmpeg_exe, "-y"]

        # Audio input BEFORE video to avoid A/V sync drift
        if cfg.include_audio and cfg.audio_device_name:
            cmd.extend([
                "-f", "dshow",
                "-thread_queue_size", "8192",
                "-async", "1",
                "-i", f"audio={cfg.audio_device_name}",
            ])

        # Video input
        cmd.extend([
            "-f", "gdigrab",
            "-framerate", str(cfg.capture_fps),
            "-offset_x", "0",
            "-offset_y", "0",
            "-video_size", f"{screen_w}x{screen_h}",
            "-use_wallclock_as_timestamps", "1",
            "-rtbufsize", "1500M",
            "-thread_queue_size", "8192" if cfg.include_audio else "512",
            "-i", "desktop",
        ])

        # Video codec
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "19",
        ])

        # Audio codec or strip
        if cfg.include_audio and cfg.audio_device_name:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", cfg.audio_bitrate,
                "-ac", str(cfg.audio_channels),
                "-ar", str(cfg.audio_sample_rate),
            ])
        else:
            cmd.append("-an")

        # Segment muxer — timestamp in filename
        cmd.extend([
            "-f", "segment",
            "-segment_time", str(cfg.segment_duration),
            "-reset_timestamps", "1",
            "-strftime", "1",
            str(cfg.buffer_dir / "%Y-%m-%d_%H-%M-%S_%%03d.ts"),
        ])

        return cmd

    def _stop_process(self) -> None:
        proc = self.process
        try:
            if proc.stdin:
                proc.stdin.write(b"q\n")
                proc.stdin.flush()
                try:
                    proc.wait(timeout=5)
                    logging.info("Desktop capture stopped cleanly")
                    return
                except subprocess.TimeoutExpired:
                    logging.warning("FFmpeg did not respond to 'q' — terminating")
                    proc.terminate()
                    proc.wait(timeout=3)
                    return
            proc.terminate()
            proc.wait(timeout=5)
            logging.info("Desktop capture stopped")
        except subprocess.TimeoutExpired:
            logging.warning("FFmpeg still alive — killing")
            proc.kill()
            proc.wait()
        except Exception as exc:
            logging.error(f"Error stopping FFmpeg: {exc}")
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


# =============================================================================
# RING BUFFER PRUNER
# =============================================================================

class RingBufferPruner:
    """
    Background daemon thread that deletes ``.ts`` segment files outside the
    rolling buffer window.

    Two safety checks are applied before any file is deleted:

    1. **Age check** — file must be older than ``config.total_buffer_seconds``
       (i.e. ``buffer_seconds + safety_seconds``).
    2. **Recency guard** — file must *also* be older than
       ``segment_duration + 2`` seconds, even if the age check passes.
       This prevents deleting a segment that FFmpeg is still writing.

    Only ``.ts`` files inside ``config.buffer_dir`` are ever touched.

    Example::

        pruner = RingBufferPruner(config)
        pruner.start()
        ...
        pruner.stop()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the pruning thread.  No-op if already running."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="RingBufferPruner")
        self.thread.start()
        logging.info("Ring buffer pruner started")

    def stop(self) -> None:
        """Signal the pruning thread to stop and join it (≤2 s wait)."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self.running:
            try:
                self._prune_once()
            except Exception as exc:
                logging.error(f"RingBufferPruner error: {exc}")
            time.sleep(1)

    def _prune_once(self) -> None:
        if self.config.dry_run:
            return

        now = time.time()

        # Oldest timestamp allowed to survive
        age_cutoff = now - self.config.total_buffer_seconds

        # Files newer than this are potentially still being written by FFmpeg
        recency_guard = now - (self.config.segment_duration + 2)

        deleted = 0
        for seg in self.config.buffer_dir.glob("*.ts"):
            try:
                mtime = seg.stat().st_mtime
                too_old = mtime < age_cutoff
                safe_to_delete = mtime < recency_guard
                if too_old and safe_to_delete:
                    seg.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass  # already removed elsewhere
            except Exception as exc:
                logging.debug(f"Could not prune {seg.name}: {exc}")

        if deleted:
            logging.debug(f"Pruned {deleted} segment(s)")


# =============================================================================
# CLI & MAIN
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffmpeg_segment_recorder",
        description=(
            "Rolling-buffer desktop recorder. "
            "Press Ctrl+C to stop. "
            "Segments are written to <out-dir>/buffer/ as .ts files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    p.add_argument(
        "--out-dir",
        default="output",
        metavar="PATH",
        help="Root output folder. Buffer segments go into <out-dir>/buffer/.",
    )

    # Capture
    p.add_argument("--fps", type=int, default=30, metavar="N", help="Capture frame rate.")
    p.add_argument(
        "--segment-seconds",
        type=int,
        default=8,
        metavar="N",
        help="Duration of each .ts segment file in seconds.",
    )

    # Buffer
    p.add_argument(
        "--buffer-seconds",
        type=int,
        default=130,
        metavar="N",
        help="How many seconds of footage to keep on disk.",
    )
    p.add_argument(
        "--safety-seconds",
        type=int,
        default=30,
        metavar="N",
        help="Extra buffer margin on top of --buffer-seconds.",
    )

    # Audio
    audio_grp = p.add_argument_group("audio (Windows DirectShow — opt-in)")
    audio_grp.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available DirectShow audio devices and exit. No recording occurs.",
    )
    audio_grp.add_argument(
        "--audio",
        action="store_true",
        help="Enable audio capture. Must be combined with --audio-device.",
    )
    audio_grp.add_argument(
        "--audio-device",
        default=None,
        metavar="NAME",
        help='DirectShow device name, e.g. "Stereo Mix (Realtek Audio)".',
    )

    # FFmpeg
    p.add_argument(
        "--ffmpeg",
        default=None,
        metavar="PATH",
        help="Path to ffmpeg.exe (falls back to imageio-ffmpeg if omitted).",
    )

    # Misc
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the FFmpeg command without executing it. No files are written.",
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_parser().parse_args(argv)

    # ------------------------------------------------------------------ config
    config = Config(
        output_base=Path(args.out_dir),
        capture_fps=args.fps,
        segment_duration=args.segment_seconds,
        buffer_seconds=args.buffer_seconds,
        safety_seconds=args.safety_seconds,
        include_audio=args.audio,
        audio_device_name=args.audio_device,
        ffmpeg_exe=args.ffmpeg,
        dry_run=args.dry_run,
    )

    # Resolve FFmpeg first (needed for --list-audio-devices too)
    try:
        config.ffmpeg_exe = find_ffmpeg(config)
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    # -------------------------------------------------------- list-audio-devices
    if args.list_audio_devices:
        list_audio_devices(config.ffmpeg_exe)
        sys.exit(0)

    # ------------------------------------------------ audio validation
    if args.audio and not args.audio_device:
        print(
            "\nERROR: --audio requires --audio-device \"<device name>\".\n"
            "       Run --list-audio-devices to see available devices.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --------------------------------------------------------- dry-run warning
    if config.dry_run:
        print(
            "\n" + "=" * 60 +
            "\n  DRY-RUN MODE — FFmpeg will NOT be executed.\n"
            "  No segments will be recorded or written to disk.\n" +
            "=" * 60 + "\n"
        )

    config.create_directories()

    # ------------------------------------------------------------------ run
    recorder = SegmentRecorder(config)
    pruner = RingBufferPruner(config)

    recorder.start()
    pruner.start()

    try:
        logging.info(
            "Recording to %s  |  buffer=%ds  |  Press Ctrl+C to stop",
            config.buffer_dir,
            config.buffer_seconds,
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pruner.stop()
        recorder.stop()
        elapsed = recorder.get_elapsed_seconds()
        logging.info("Session ended after %.0f seconds.", elapsed)


if __name__ == "__main__":
    main()
