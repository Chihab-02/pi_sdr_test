#!/usr/bin/env python3
"""
USRP B210 desktop app — FFT on a chosen receive path.

  Pick channel A or B and RX2 (dedicated) or TX/RX (shared jack as RX input).
  Default: Channel A + TX/RX as receiver. Modes: RX-only or TX/RX loopback + RX FFT.

  python3 b210_fft.py [--args type=b200]
"""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gnuradio import analog, blocks, fft, gr, qtgui, uhd
from gnuradio.fft import window

CH_A = 0
CH_B = 1

# Brief pause after releasing the USRP before opening again (avoids USB/UHD bus errors).
_USRP_REOPEN_DELAY_S = 0.35


@dataclass
class RunConfig:
    device_args: str
    channel: int          # physical UHD channel (0 = A, 1 = B)
    mode: str             # "rx" | "txrx"
    rx_antenna: str       # "RX2" or "TX/RX"
    center_hz: float
    sample_rate: float
    rx_gain_db: float
    fft_size: int
    tx_gain_db: float
    tx_offset_hz: float
    tx_ampl: float
    expected_peak_hz: float
    peak_use_max_hold: bool
    peak_update_stride: int


def _ch_name(idx: int) -> str:
    return "A" if idx == CH_A else "B"


def _rx_label(idx: int) -> str:
    return "RX1 (RF1)" if idx == CH_A else "RX2 (RF2)"


def _tx_label(idx: int) -> str:
    return "TX1 (RF1)" if idx == CH_A else "TX2 (RF2)"


def _wrap_freq_sink_widget(sink: qtgui.freq_sink_c) -> QWidget:
    return sip.wrapinstance(int(sink.qwidget()), QWidget)


class PeakEmitter(QObject):
    """Thread-safe delivery of peak-analysis text from the GR worker to the GUI."""
    text_ready = pyqtSignal(str)


def _apply_rx_antenna(src: uhd.usrp_source, name: str) -> None:
    """
    Select the physical RX path on GR channel 0.
    The GR UHD block's internal channel index is always 0 regardless of which
    physical channel was selected in stream_args.channels.
    """
    if not name:
        return
    available = src.get_antennas(0)
    if name not in available:
        raise RuntimeError(
            f"RX antenna '{name}' is not available on this channel.\n"
            f"UHD reports: {', '.join(available)}"
        )
    src.set_antenna(name, 0)


def _interp_freq_at_level(
    freqs_hz: np.ndarray, mag_db: np.ndarray, i0: int, i1: int, level_db: float
) -> float:
    y0, y1 = float(mag_db[i0]), float(mag_db[i1])
    f0, f1 = float(freqs_hz[i0]), float(freqs_hz[i1])
    if y1 == y0:
        return 0.5 * (f0 + f1)
    t = float(np.clip((level_db - y0) / (y1 - y0), 0.0, 1.0))
    return f0 + t * (f1 - f0)


def _parabolic_peak_freq(
    freqs_hz: np.ndarray, mag_db: np.ndarray, p: int
) -> tuple[float, float]:
    if p <= 0 or p >= len(mag_db) - 1:
        return float(freqs_hz[p]), float(mag_db[p])
    y0, y1, y2 = float(mag_db[p - 1]), float(mag_db[p]), float(mag_db[p + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs_hz[p]), y1
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    df = float(freqs_hz[p + 1] - freqs_hz[p])
    f_peak = float(freqs_hz[p]) + delta * df
    y_peak = y1 - 0.25 * (y0 - y2) * delta
    return f_peak, y_peak


def _edges_3db_hz(
    freqs_hz: np.ndarray, mag_db: np.ndarray, peak_idx: int, peak_db: float
) -> tuple[float, float, float]:
    target = peak_db - 3.0
    n = len(mag_db)

    j = peak_idx
    while j > 0 and mag_db[j] > target:
        j -= 1
    if j + 1 < n and mag_db[j] <= target < mag_db[j + 1]:
        f_left = _interp_freq_at_level(freqs_hz, mag_db, j, j + 1, target)
    else:
        f_left = float(freqs_hz[max(0, j)])

    j = peak_idx
    while j < n - 1 and mag_db[j] > target:
        j += 1
    if j > 0 and mag_db[j] <= target < mag_db[j - 1]:
        f_right = _interp_freq_at_level(freqs_hz, mag_db, j - 1, j, target)
    else:
        f_right = float(freqs_hz[min(n - 1, j)])

    return f_left, f_right, max(0.0, f_right - f_left)


def _snr_median_mask(
    mag_db: np.ndarray, peak_idx: int, guard_bins: int
) -> tuple[float, float]:
    n = len(mag_db)
    g = max(4, int(guard_bins))
    mask = np.ones(n, dtype=bool)
    mask[max(0, peak_idx - g) : min(n, peak_idx + g + 1)] = False
    nf = float(np.median(mag_db[mask]) if np.any(mask) else np.median(mag_db))
    return float(mag_db[peak_idx]) - nf, nf


def _build_peak_report(
    mag_lin_natural: np.ndarray,
    sample_rate: float,
    center_hz: float,
    expected_peak_hz: float,
    fft_size: int,
) -> str:
    """Natural-order |FFT| bins → shifted dB spectrum → single-line peak report."""
    n = int(mag_lin_natural.shape[0])
    mag_db = np.fft.fftshift(20.0 * np.log10(np.maximum(mag_lin_natural, 1e-20)))
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate)) + center_hz

    half = max(8, min(fft_size // 64, 64))
    i0 = int(np.argmin(np.abs(freqs_hz - expected_peak_hz)))
    lo, hi = max(0, i0 - half), min(n, i0 + half + 1)
    p = lo + int(np.argmax(mag_db[lo:hi]))
    f_c, db_c = _parabolic_peak_freq(freqs_hz, mag_db, p)
    _, _, bw = _edges_3db_hz(freqs_hz, mag_db, p, db_c)
    snr_db, _ = _snr_median_mask(mag_db, p, guard_bins=max(half, fft_size // 32))

    return (
        f"Peak: {f_c / 1e6:.6f} MHz  |  {db_c:+.1f} dB  |  "
        f"SNR {snr_db:.1f} dB  |  BW {bw / 1e3:.2f} kHz"
    )


class _FftPeakAnalyzer(gr.sync_block):
    """
    Parallel FFT path — does not drive the display.
    Operates on complex vectors (output of fft.fft_vcc), computes |FFT|,
    optionally maintains a running max-hold, and calls back every N vectors.
    """

    def __init__(
        self,
        fft_size: int,
        sample_rate: float,
        center_hz: float,
        expected_peak_hz: float,
        use_max_hold: bool,
        update_stride: int,
        callback: Callable[[str], None],
    ) -> None:
        gr.sync_block.__init__(
            self,
            "fft_peak_analyzer",
            [(np.complex64, fft_size)],
            None,
        )
        self._fft_size = fft_size
        self._sample_rate = sample_rate
        self._center_hz = center_hz
        self._expected_peak_hz = float(expected_peak_hz)
        self._use_max_hold = bool(use_max_hold)
        self._stride = max(1, int(update_stride))
        self._callback = callback
        self._max_lin = np.zeros(fft_size, dtype=np.float64)
        self._vcounter = 0

    def work(self, input_items, output_items) -> int:
        inv = input_items[0]
        for vec in inv:
            self._vcounter += 1
            cur = np.abs(np.asarray(vec))
            if self._use_max_hold:
                self._max_lin = np.maximum(self._max_lin, cur)
                mag = self._max_lin
            else:
                mag = cur

            if self._vcounter % self._stride != 0:
                continue
            try:
                text = _build_peak_report(
                    mag,
                    self._sample_rate,
                    self._center_hz,
                    self._expected_peak_hz,
                    self._fft_size,
                )
                self._callback(text)
            except Exception as ex:
                self._callback(f"Peak analysis error: {ex!r}")
        return len(inv)


class B210Flowgraph(gr.top_block):
    """
    RX → FFT display; optional TX test tone on the same channel (txrx mode).

    Important: all UHD set_*/get_* calls use GR channel index 0 because each
    source/sink block is created with a single-element channels list.  The
    physical channel selection is handled entirely through stream_args.channels.
    """

    def __init__(
        self,
        cfg: RunConfig,
        peak_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        gr.top_block.__init__(self, "B210 RX / TX-RX")
        self._fft_widget: Optional[QWidget] = None
        self._freq_sink: Optional[qtgui.freq_sink_c] = None

        # ── RX source ────────────────────────────────────────────────────────
        rx_stream = uhd.stream_args(
            cpu_format="fc32",
            otw_format="sc16",
            channels=[cfg.channel],          # physical channel selection
        )
        self.src = uhd.usrp_source(cfg.device_args, rx_stream)
        self.src.set_samp_rate(cfg.sample_rate)
        self.src.set_center_freq(cfg.center_hz, 0)   # GR channel index = 0
        self.src.set_gain(cfg.rx_gain_db, 0)          # GR channel index = 0
        _apply_rx_antenna(self.src, cfg.rx_antenna)

        # ── TX source (loopback mode only) ───────────────────────────────────
        if cfg.mode == "txrx":
            tx_stream = uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                channels=[cfg.channel],      # same physical channel
            )
            self.snk = uhd.usrp_sink(cfg.device_args, tx_stream, "")
            self.snk.set_samp_rate(cfg.sample_rate)
            self.snk.set_center_freq(cfg.center_hz, 0)  # GR channel index = 0
            self.snk.set_gain(cfg.tx_gain_db, 0)         # GR channel index = 0
            self.tx_src = analog.sig_source_c(
                cfg.sample_rate,
                analog.GR_COS_WAVE,
                cfg.tx_offset_hz,
                cfg.tx_ampl,
                0,
            )
            self.connect(self.tx_src, self.snk)

        # ── FFT display sink ─────────────────────────────────────────────────
        ch = cfg.channel
        ant_note = cfg.rx_antenna or "?"
        if cfg.mode == "rx":
            title = f"FFT RX — ch {_ch_name(ch)} — {ant_note} — {_rx_label(ch)}"
        else:
            title = (
                f"TX/RX loopback (RX FFT) — ch {_ch_name(ch)} "
                f"— RX port {ant_note} ({_tx_label(ch)} → RX)"
            )

        sink = qtgui.freq_sink_c(
            cfg.fft_size,
            window.WIN_BLACKMAN_hARRIS,
            cfg.center_hz,
            cfg.sample_rate,
            title,
            1,
            None,
        )
        self._freq_sink = sink
        sink.set_update_time(0.10)
        sink.set_y_axis(-140, 10)
        self.connect((self.src, 0), (sink, 0))
        self._fft_widget = _wrap_freq_sink_widget(sink)

        # ── Parallel peak-analysis path ───────────────────────────────────────
        if peak_callback is not None:
            win_taps = window.build(window.WIN_BLACKMAN_hARRIS, cfg.fft_size)
            self._stv = blocks.stream_to_vector(gr.sizeof_gr_complex, cfg.fft_size)
            self._fft_vcc = fft.fft_vcc(cfg.fft_size, True, win_taps, False, 1)
            self._peak_an = _FftPeakAnalyzer(
                cfg.fft_size,
                cfg.sample_rate,
                cfg.center_hz,
                cfg.expected_peak_hz,
                cfg.peak_use_max_hold,
                cfg.peak_update_stride,
                peak_callback,
            )
            self.connect((self.src, 0), (self._stv, 0))
            self.connect((self._stv, 0), (self._fft_vcc, 0))
            self.connect((self._fft_vcc, 0), (self._peak_an, 0))


class MainWindow(QWidget):
    def __init__(self, default_device_args: str) -> None:
        super().__init__()
        self._tb: Optional[B210Flowgraph] = None
        self._peak_emitter = PeakEmitter()
        self._peak_emitter.text_ready.connect(self._on_peak_text)

        # Keep the Qt event loop alive so GR callbacks can reach the GUI.
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: None)
        self._timer.start(100)

        self.setWindowTitle("USRP B210 — receiver (FFT)")
        self.resize(820, 500)

        small_font = QFont()
        small_font.setPointSize(8)
        self.setFont(small_font)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        intro = QLabel(
            "<b>RX only</b>: external signal &nbsp;·&nbsp; "
            "<b>TX/RX</b>: internal tone loopback + FFT"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # ── Mode / Port collapsible section ──────────────────────────────────
        self._mode_port_summary = QLabel()
        self._mode_port_summary.setWordWrap(True)

        self._btn_mode_port = QPushButton("Mode/Port…")
        self._btn_mode_port.setCheckable(True)
        self._btn_mode_port.setChecked(False)
        self._btn_mode_port.toggled.connect(self._on_mode_port_toggled)

        mode_box = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setContentsMargins(4, 4, 4, 4)
        mode_layout.setSpacing(1)
        self._rb_rx_only = QRadioButton("RX only")
        self._rb_txrx = QRadioButton("TX/RX test")
        self._rb_rx_only.setChecked(True)
        self._grp_mode = QButtonGroup(self)
        self._grp_mode.addButton(self._rb_rx_only)
        self._grp_mode.addButton(self._rb_txrx)
        mode_layout.addWidget(self._rb_rx_only)
        mode_layout.addWidget(self._rb_txrx)

        port_box = QGroupBox("RX Port")
        port_layout = QVBoxLayout(port_box)
        port_layout.setContentsMargins(4, 4, 4, 4)
        port_layout.setSpacing(1)
        self._grp_port = QButtonGroup(self)
        self._rb_a_rx2 = QRadioButton("A — RX2")
        self._rb_a_txrx = QRadioButton("A — TX/RX")
        self._rb_b_rx2 = QRadioButton("B — RX2")
        self._rb_b_txrx = QRadioButton("B — TX/RX")
        self._rb_a_txrx.setChecked(True)
        for rb in (self._rb_a_rx2, self._rb_a_txrx, self._rb_b_rx2, self._rb_b_txrx):
            self._grp_port.addButton(rb)
        port_layout.addWidget(self._rb_a_rx2)
        port_layout.addWidget(self._rb_a_txrx)
        port_layout.addWidget(self._rb_b_rx2)
        port_layout.addWidget(self._rb_b_txrx)

        self._mode_port_panel = QWidget()
        mp_layout = QVBoxLayout(self._mode_port_panel)
        mp_layout.setContentsMargins(0, 0, 0, 0)
        mp_layout.setSpacing(2)
        mp_layout.addWidget(mode_box)
        mp_layout.addWidget(port_box)
        self._mode_port_panel.setVisible(False)

        root.addWidget(self._mode_port_summary)
        root.addWidget(self._btn_mode_port)
        root.addWidget(self._mode_port_panel)

        for rb in (
            self._rb_rx_only, self._rb_txrx,
            self._rb_a_rx2, self._rb_a_txrx,
            self._rb_b_rx2, self._rb_b_txrx,
        ):
            rb.toggled.connect(self._update_mode_port_summary)

        self._update_mode_port_summary()

        # ── Tuning ───────────────────────────────────────────────────────────
        main_box = QGroupBox("Tuning")
        main_form = QFormLayout(main_box)
        main_form.setContentsMargins(4, 4, 4, 4)
        main_form.setSpacing(2)

        self._freq_mhz = QDoubleSpinBox()
        self._freq_mhz.setRange(1.0, 6000.0)
        self._freq_mhz.setDecimals(3)
        self._freq_mhz.setValue(100.0)
        self._freq_mhz.setSuffix(" MHz")
        main_form.addRow("Freq:", self._freq_mhz)

        self._rx_gain = QDoubleSpinBox()
        self._rx_gain.setRange(0.0, 76.0)
        self._rx_gain.setDecimals(1)
        self._rx_gain.setValue(40.0)
        self._rx_gain.setSuffix(" dB")
        main_form.addRow("RX Gain:", self._rx_gain)
        root.addWidget(main_box)

        # ── Advanced collapsible section ──────────────────────────────────────
        self._btn_advanced = QPushButton("Advanced…")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setChecked(False)
        self._btn_advanced.toggled.connect(self._on_advanced_toggled)
        root.addWidget(self._btn_advanced)

        self._advanced = QWidget()
        adv_layout = QVBoxLayout(self._advanced)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(2)
        adv_box = QGroupBox("Advanced")
        form = QFormLayout(adv_box)
        form.setContentsMargins(4, 4, 4, 4)
        form.setSpacing(2)

        self._device_args = QLineEdit(default_device_args)
        form.addRow("Args:", self._device_args)

        self._rate_msps = QDoubleSpinBox()
        self._rate_msps.setRange(0.25, 40.0)
        self._rate_msps.setDecimals(3)
        self._rate_msps.setValue(4.0)
        self._rate_msps.setSuffix(" Msps")
        form.addRow("Rate:", self._rate_msps)

        self._fft = QSpinBox()
        self._fft.setRange(256, 8192)
        self._fft.setSingleStep(256)
        self._fft.setValue(1024)
        form.addRow("FFT size:", self._fft)

        self._peak_max_hold = QCheckBox("Max-hold")
        self._peak_max_hold.setChecked(True)
        self._peak_max_hold.setToolTip("Running max-hold per bin for peak readout")
        form.addRow("Peak:", self._peak_max_hold)

        self._peak_stride = QSpinBox()
        self._peak_stride.setRange(1, 128)
        self._peak_stride.setValue(8)
        self._peak_stride.setToolTip("Update peak readout every N FFT vectors")
        form.addRow("Stride:", self._peak_stride)

        self._tx_gain = QDoubleSpinBox()
        self._tx_gain.setRange(0.0, 90.0)
        self._tx_gain.setDecimals(1)
        self._tx_gain.setValue(20.0)
        self._tx_gain.setSuffix(" dB")
        form.addRow("TX Gain:", self._tx_gain)

        self._tx_off_khz = QDoubleSpinBox()
        self._tx_off_khz.setRange(1.0, 5000.0)
        self._tx_off_khz.setDecimals(1)
        self._tx_off_khz.setValue(100.0)
        self._tx_off_khz.setSuffix(" kHz")
        form.addRow("TX Offset:", self._tx_off_khz)

        self._tx_ampl = QDoubleSpinBox()
        self._tx_ampl.setRange(0.01, 1.0)
        self._tx_ampl.setDecimals(3)
        self._tx_ampl.setValue(0.2)
        self._tx_ampl.setSingleStep(0.05)
        form.addRow("TX Ampl:", self._tx_ampl)

        adv_layout.addWidget(adv_box)
        self._advanced.setVisible(False)
        root.addWidget(self._advanced)

        self._rb_rx_only.toggled.connect(self._sync_tx_controls)
        self._rb_txrx.toggled.connect(self._sync_tx_controls)

        # ── Start / Stop ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_start = QPushButton("Start")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        # ── Peak readout ──────────────────────────────────────────────────────
        peak_box = QGroupBox("Peak")
        peak_layout = QVBoxLayout(peak_box)
        peak_layout.setContentsMargins(4, 4, 4, 4)
        peak_layout.setSpacing(1)
        self._peak_view = QLabel("Idle")
        self._peak_view.setFont(QFont("monospace", 7))
        peak_layout.addWidget(self._peak_view)
        root.addWidget(peak_box)

        # ── FFT display area ──────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._placeholder = QLabel("Press  Start")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._stack.addWidget(self._placeholder)
        root.addWidget(self._stack, stretch=1)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._sync_tx_controls()

    # ── Signal routing ────────────────────────────────────────────────────────

    def _dispatch_peak(self, text: str) -> None:
        """Called from GR worker thread — must NOT touch Qt widgets directly."""
        print(text, file=sys.stderr, flush=True)
        self._peak_emitter.text_ready.emit(text)

    def _on_peak_text(self, text: str) -> None:
        self._peak_view.setText(text)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _on_advanced_toggled(self, on: bool) -> None:
        self._advanced.setVisible(on)
        self._btn_advanced.setText("Hide advanced" if on else "Advanced…")

    def _on_mode_port_toggled(self, on: bool) -> None:
        self._mode_port_panel.setVisible(on)
        self._btn_mode_port.setText("Hide…" if on else "Mode/Port…")

    def _update_mode_port_summary(self) -> None:
        mode = "TX/RX" if self._rb_txrx.isChecked() else "RX"
        if self._rb_a_rx2.isChecked():
            port = "A — RX2"
        elif self._rb_a_txrx.isChecked():
            port = "A — TX/RX"
        elif self._rb_b_rx2.isChecked():
            port = "B — RX2"
        else:
            port = "B — TX/RX"
        self._mode_port_summary.setText(f"<b>{mode}</b>  ·  {port}")

    def _sync_tx_controls(self) -> None:
        tx_on = self._rb_txrx.isChecked()
        self._tx_gain.setEnabled(tx_on)
        self._tx_ampl.setEnabled(tx_on)
        self._tx_off_khz.setEnabled(True)   # also controls expected peak window

    def _selected_rx_port(self) -> tuple[int, str]:
        """Returns (physical UHD channel index, RX antenna name)."""
        if self._rb_a_rx2.isChecked():
            return CH_A, "RX2"
        if self._rb_a_txrx.isChecked():
            return CH_A, "TX/RX"
        if self._rb_b_rx2.isChecked():
            return CH_B, "RX2"
        if self._rb_b_txrx.isChecked():
            return CH_B, "TX/RX"
        return CH_A, "TX/RX"

    def _selected_mode(self) -> str:
        return "txrx" if self._rb_txrx.isChecked() else "rx"

    def _set_running_state(self, running: bool) -> None:
        """Lock controls while the flowgraph owns the USRP."""
        for w in (
            self._rb_a_rx2, self._rb_a_txrx, self._rb_b_rx2, self._rb_b_txrx,
            self._rb_rx_only, self._rb_txrx,
            self._btn_mode_port, self._freq_mhz, self._rx_gain,
            self._btn_advanced, self._advanced,
            self._device_args, self._rate_msps, self._fft,
            self._peak_max_hold, self._peak_stride,
            self._tx_gain, self._tx_off_khz, self._tx_ampl,
        ):
            w.setEnabled(not running)
        if not running:
            self._sync_tx_controls()

    def _build_config(self) -> RunConfig:
        ch, ant = self._selected_rx_port()
        center_hz = self._freq_mhz.value() * 1e6
        tone_off_hz = self._tx_off_khz.value() * 1e3
        return RunConfig(
            device_args=self._device_args.text().strip(),
            channel=ch,
            mode=self._selected_mode(),
            rx_antenna=ant,
            center_hz=center_hz,
            sample_rate=self._rate_msps.value() * 1e6,
            rx_gain_db=self._rx_gain.value(),
            fft_size=self._fft.value(),
            tx_gain_db=self._tx_gain.value(),
            tx_offset_hz=tone_off_hz,
            tx_ampl=self._tx_ampl.value(),
            expected_peak_hz=center_hz + tone_off_hz,
            peak_use_max_hold=self._peak_max_hold.isChecked(),
            peak_update_stride=int(self._peak_stride.value()),
        )

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._tb is not None:
            self._on_stop()
            time.sleep(_USRP_REOPEN_DELAY_S)
            gc.collect()
            QApplication.processEvents()

        cfg = self._build_config()
        try:
            self._tb = B210Flowgraph(cfg, peak_callback=self._dispatch_peak)
        except Exception as e:
            QMessageBox.critical(self, "UHD / GNU Radio error", str(e))
            self._tb = None
            return

        # Remove any previous FFT widget before adding the new one.
        self._remove_fft_widget()

        assert self._tb._fft_widget is not None
        self._stack.addWidget(self._tb._fft_widget)
        self._stack.setCurrentIndex(1)

        c = _ch_name(cfg.channel)
        port = cfg.rx_antenna
        if cfg.mode == "rx":
            self.setWindowTitle(f"USRP B210 — FFT RX — ch {c} — {port}")
        else:
            self.setWindowTitle(f"USRP B210 — loopback — ch {c} — RX {port}")

        self._peak_view.setText("Running…")
        self._tb.start()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_running_state(True)

    def _on_stop(self) -> None:
        if self._tb is None:
            return
        self._tb.stop()
        self._tb.wait()

        # Hide and explicitly release the sip-wrapped Qt widget BEFORE the
        # flowgraph (and its freq_sink) is garbage-collected.  Reversing this
        # order can produce a segfault because the C++ freq_sink destructor
        # would try to destroy a widget that PyQt has already cleaned up.
        self._remove_fft_widget()

        self._tb = None
        gc.collect()
        QApplication.processEvents()

        self._stack.setCurrentIndex(0)
        self.setWindowTitle("USRP B210 — receiver (FFT)")
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_running_state(False)
        self._peak_view.setText("Idle")

    def _remove_fft_widget(self) -> None:
        """Safely remove and delete any sip-wrapped FFT widget from the stack."""
        while self._stack.count() > 1:
            w = self._stack.widget(1)
            w.hide()
            self._stack.removeWidget(w)
            sip.delete(w)

    def closeEvent(self, event) -> None:
        self._on_stop()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B210: RX spectrum; optional TX/RX loopback viewed on RX FFT",
    )
    parser.add_argument(
        "--args",
        default="type=b200",
        help="Default UHD device args string (editable at runtime under Advanced).",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(default_device_args=args.args)
    win.show()

    signal.signal(signal.SIGINT, lambda _s, _f: win.close())

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
