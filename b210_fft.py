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
from PyQt5 import Qt, sip
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
_USRP_REOPEN_DELAY_S = 0.25


@dataclass
class RunConfig:
    device_args: str
    channel: int
    mode: str  # "rx" | "txrx"
    """UHD RX antenna: 'RX2' (dedicated) or 'TX/RX' (shared). Always set for port tests."""
    rx_antenna: str
    center_hz: float
    sample_rate: float
    rx_gain_db: float
    fft_size: int
    tx_gain_db: float
    tx_offset_hz: float
    tx_ampl: float
    """Absolute RF frequency of the default tone (center + TX offset); peak search is anchored here."""
    expected_peak_hz: float
    """If True, maintain running per-bin max |FFT| (parallel path only); else use each FFT frame alone."""
    peak_use_max_hold: bool
    """Emit peak/SNR readout every N FFT vectors (throttle CPU; spectrum FFT stays real-time)."""
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
    """Thread-safe delivery of peak-analysis text from the GNU Radio worker to the GUI."""

    text_ready = pyqtSignal(str)


def _apply_rx_antenna(src: uhd.usrp_source, channel: int, name: str) -> None:
    """Select which physical RX path to use (B210: 'RX2' dedicated, 'TX/RX' shared)."""
    if not name:
        return
    available = src.get_antennas(channel)
    if name not in available:
        raise RuntimeError(
            f"RX antenna '{name}' is not available on this channel. "
            f"UHD reports: {', '.join(available)}"
        )
    src.set_antenna(name, channel)


def _interp_freq_at_level(
    freqs_hz: np.ndarray, mag_db: np.ndarray, i0: int, i1: int, level_db: float
) -> float:
    """Linear interpolation of frequency where mag_db crosses level_db between bins i0 and i1."""
    y0, y1 = float(mag_db[i0]), float(mag_db[i1])
    f0, f1 = float(freqs_hz[i0]), float(freqs_hz[i1])
    if y1 == y0:
        return 0.5 * (f0 + f1)
    t = (level_db - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return f0 + t * (f1 - f0)


def _parabolic_peak_freq(
    freqs_hz: np.ndarray, mag_db: np.ndarray, p: int
) -> tuple[float, float]:
    """Refine peak bin index with parabolic fit on dB magnitudes; return (freq_hz, peak_db)."""
    if p <= 0 or p >= len(mag_db) - 1:
        return float(freqs_hz[p]), float(mag_db[p])
    y0, y1, y2 = float(mag_db[p - 1]), float(mag_db[p]), float(mag_db[p + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs_hz[p]), y1
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    df = float(freqs_hz[p + 1] - freqs_hz[p])
    f_peak = float(freqs_hz[p]) + delta * df
    y_peak = y1 - 0.25 * (y0 - y2) * delta
    return f_peak, y_peak


def _edges_3db_hz(
    freqs_hz: np.ndarray, mag_db: np.ndarray, peak_idx: int, peak_db: float
) -> tuple[float, float, float]:
    """
    Half-power (-3 dB) edges relative to peak_db (interpolated between bins).
    Returns (f_left_hz, f_right_hz, width_hz).
    """
    target = peak_db - 3.0
    n = len(mag_db)

    # Left: move left from peak until magnitude is at/below target; crossing is (j, j+1).
    j = peak_idx
    while j > 0 and mag_db[j] > target:
        j -= 1
    if j + 1 < n and mag_db[j] <= target < mag_db[j + 1]:
        f_left = _interp_freq_at_level(freqs_hz, mag_db, j, j + 1, target)
    else:
        f_left = float(freqs_hz[max(0, j)])

    # Right: move right from peak until magnitude is at/below target; crossing is (j-1, j).
    j = peak_idx
    while j < n - 1 and mag_db[j] > target:
        j += 1
    if j > 0 and mag_db[j] <= target < mag_db[j - 1]:
        f_right = _interp_freq_at_level(freqs_hz, mag_db, j - 1, j, target)
    else:
        f_right = float(freqs_hz[min(n - 1, j)])

    width = max(0.0, f_right - f_left)
    return f_left, f_right, width


def _snr_median_mask(
    mag_db: np.ndarray, peak_idx: int, guard_bins: int
) -> tuple[float, float]:
    """SNR vs median noise: noise = median of bins outside ±guard_bins around peak_idx."""
    n = len(mag_db)
    g = max(4, int(guard_bins))
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_idx - g)
    hi = min(n, peak_idx + g + 1)
    mask[lo:hi] = False
    if not np.any(mask):
        nf = float(np.median(mag_db))
    else:
        nf = float(np.median(mag_db[mask]))
    pk = float(mag_db[peak_idx])
    return pk - nf, nf


def _build_peak_report_from_mag_lin(
    mag_lin_natural: np.ndarray,
    sample_rate: float,
    center_hz: float,
    expected_peak_hz: float,
    fft_size: int,
) -> str:
    """Natural-order |FFT| bins → shifted dB spectrum → single-line peak report."""
    n = int(mag_lin_natural.shape[0])
    mag_db = 20.0 * np.log10(np.maximum(mag_lin_natural, 1e-20))
    mag_db = np.fft.fftshift(mag_db)
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate)) + center_hz

    half = max(8, min(fft_size // 64, 64))
    i0 = int(np.argmin(np.abs(freqs_hz - expected_peak_hz)))
    lo = max(0, i0 - half)
    hi = min(n, i0 + half + 1)
    seg = mag_db[lo:hi]
    rel = int(np.argmax(seg))
    p = lo + rel
    f_c, db_c = _parabolic_peak_freq(freqs_hz, mag_db, p)
    _, _, bw = _edges_3db_hz(freqs_hz, mag_db, p, db_c)
    snr_db, _ = _snr_median_mask(mag_db, p, guard_bins=max(half, fft_size // 32))

    return (
        f"Peak: {f_c/1e6:.6f} MHz | {db_c:+.1f} dB | SNR {snr_db:.1f} dB | BW {bw/1e3:.2f} kHz"
    )


class _FftPeakAnalyzer(gr.sync_block):
    """
    Parallel FFT path (same window/size as qtgui.freq_sink) — does not drive the display.
    Collects |FFT| bins each vector; optional running max-hold; real-time readout every N frames.
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
        n = len(inv)
        for i in range(n):
            self._vcounter += 1
            spec = np.asarray(inv[i])
            cur = np.abs(spec)
            if self._use_max_hold:
                self._max_lin = np.maximum(self._max_lin, cur)
                mag_for_report = self._max_lin
            else:
                mag_for_report = cur

            if self._vcounter % self._stride != 0:
                continue
            try:
                text = _build_peak_report_from_mag_lin(
                    mag_for_report,
                    self._sample_rate,
                    self._center_hz,
                    self._expected_peak_hz,
                    self._fft_size,
                )
                self._callback(text)
            except Exception as ex:
                self._callback(f"Peak analysis error: {ex!r}")
        return n


class B210Flowgraph(gr.top_block):
    """RX → FFT; optional TX test tone on same channel (txrx)."""

    def __init__(
        self,
        cfg: RunConfig,
        peak_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        gr.top_block.__init__(self, "B210 RX / TX-RX")
        self._fft_widget: QWidget | None = None
        self._freq_sink: qtgui.freq_sink_c | None = None
        ch = cfg.channel

        rx_stream = uhd.stream_args(
            cpu_format="fc32",
            otw_format="sc16",
            channels=[ch],
        )
        self.src = uhd.usrp_source(cfg.device_args, rx_stream, True)
        self.src.set_samp_rate(cfg.sample_rate)
        self.src.set_center_freq(cfg.center_hz, ch)
        self.src.set_gain(cfg.rx_gain_db, ch)
        _apply_rx_antenna(self.src, ch, cfg.rx_antenna)

        if cfg.mode == "txrx":
            tx_stream = uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                channels=[ch],
            )
            self.snk = uhd.usrp_sink(cfg.device_args, tx_stream, "")
            self.snk.set_samp_rate(cfg.sample_rate)
            self.snk.set_center_freq(cfg.center_hz, ch)
            self.snk.set_gain(cfg.tx_gain_db, ch)
            self.tx_src = analog.sig_source_c(
                cfg.sample_rate,
                analog.GR_COS_WAVE,
                cfg.tx_offset_hz,
                cfg.tx_ampl,
                0,
            )
            self.connect(self.tx_src, self.snk)

        ant_note = cfg.rx_antenna or "?"
        if cfg.mode == "rx":
            title = f"FFT RX — ch {_ch_name(ch)} — {ant_note} — {_rx_label(ch)}"
        else:
            title = (
                f"TX/RX loopback (RX FFT) — ch {_ch_name(ch)} — RX port {ant_note} "
                f"({_tx_label(ch)} → RX)"
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

        if peak_callback is not None:
            win = window.build(window.WIN_BLACKMAN_hARRIS, cfg.fft_size)
            self._stv = blocks.stream_to_vector(gr.sizeof_gr_complex, cfg.fft_size)
            self._fft_vcc = fft.fft_vcc(cfg.fft_size, True, win, False, 1)
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
        self._tb: B210Flowgraph | None = None
        self._peak_emitter = PeakEmitter()
        self._peak_emitter.text_ready.connect(self._on_peak_text)
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: None)
        self._timer.start(100)

        self.setWindowTitle("USRP B210 — receiver (FFT)")
        self.resize(900, 900)

        root = QVBoxLayout(self)

        intro = QLabel(
            "<b>FFT on every receive-capable jack.</b> The line below shows the active mode and RX port; "
            "use <b>Show different mode &amp; receive port…</b> to change them (same idea as "
            "<b>Show advanced parameters…</b>). "
            "Channel A/B × RX2 or TX/RX selects the UHD RX path — TX/RX can be receive-only on that jack.<br>"
            "• <b>RX only</b>: external signal into the selected connector.<br>"
            "• <b>TX/RX test</b>: internal tone + FFT on the selected RX port (use cable/attenuator as needed)."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._mode_port_summary = QLabel()
        self._mode_port_summary.setWordWrap(True)

        self._btn_mode_port = QPushButton("Show different mode & receive port…")
        self._btn_mode_port.setCheckable(True)
        self._btn_mode_port.setChecked(False)
        self._btn_mode_port.toggled.connect(self._on_mode_port_toggled)

        mode_box = QGroupBox("Mode (both show the RX spectrum)")
        mode_layout = QVBoxLayout(mode_box)
        self._rb_rx_only = QRadioButton("RX only — no transmission from the B210")
        self._rb_txrx = QRadioButton(
            "TX/RX test — TX tone + RX spectrum (validation on receive path)"
        )
        self._rb_rx_only.setChecked(True)
        self._grp_mode = QButtonGroup(self)
        self._grp_mode.addButton(self._rb_rx_only)
        self._grp_mode.addButton(self._rb_txrx)
        mode_layout.addWidget(self._rb_rx_only)
        mode_layout.addWidget(self._rb_txrx)

        port_box = QGroupBox("Receive port (FFT uses this RX path)")
        port_layout = QVBoxLayout(port_box)
        self._grp_port = QButtonGroup(self)
        self._rb_a_rx2 = QRadioButton(
            "Channel A — RX2 (dedicated receive jack on RF chain A)"
        )
        self._rb_a_txrx = QRadioButton(
            "Channel A — TX/RX (shared jack — receive on this connector)"
        )
        self._rb_b_rx2 = QRadioButton(
            "Channel B — RX2 (dedicated receive jack on RF chain B)"
        )
        self._rb_b_txrx = QRadioButton(
            "Channel B — TX/RX (shared jack — receive on this connector)"
        )
        self._rb_a_txrx.setChecked(True)
        for rb in (
            self._rb_a_rx2,
            self._rb_a_txrx,
            self._rb_b_rx2,
            self._rb_b_txrx,
        ):
            self._grp_port.addButton(rb)
        port_layout.addWidget(self._rb_a_rx2)
        port_layout.addWidget(self._rb_a_txrx)
        port_layout.addWidget(self._rb_b_rx2)
        port_layout.addWidget(self._rb_b_txrx)

        self._mode_port_panel = QWidget()
        mp_layout = QVBoxLayout(self._mode_port_panel)
        mp_layout.setContentsMargins(0, 0, 0, 0)
        mp_layout.addWidget(mode_box)
        mp_layout.addWidget(port_box)
        self._mode_port_panel.setVisible(False)

        root.addWidget(self._mode_port_summary)
        root.addWidget(self._btn_mode_port)
        root.addWidget(self._mode_port_panel)

        for rb in (
            self._rb_rx_only,
            self._rb_txrx,
            self._rb_a_rx2,
            self._rb_a_txrx,
            self._rb_b_rx2,
            self._rb_b_txrx,
        ):
            rb.toggled.connect(self._update_mode_port_summary)

        self._update_mode_port_summary()

        # Essential controls (always visible)
        main_box = QGroupBox("Tuning")
        main_form = QFormLayout(main_box)
        self._freq_mhz = QDoubleSpinBox()
        self._freq_mhz.setRange(1.0, 6000.0)
        self._freq_mhz.setDecimals(3)
        self._freq_mhz.setValue(100.0)
        self._freq_mhz.setSuffix(" MHz")
        main_form.addRow("Center frequency:", self._freq_mhz)

        self._rx_gain = QDoubleSpinBox()
        self._rx_gain.setRange(0.0, 76.0)
        self._rx_gain.setDecimals(1)
        self._rx_gain.setValue(40.0)
        self._rx_gain.setSuffix(" dB")
        main_form.addRow("RX gain:", self._rx_gain)
        root.addWidget(main_box)

        # Advanced parameters (hidden by default)
        self._btn_advanced = QPushButton("Show advanced parameters…")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setChecked(False)
        self._btn_advanced.toggled.connect(self._on_advanced_toggled)
        root.addWidget(self._btn_advanced)

        self._advanced = QWidget()
        adv_layout = QVBoxLayout(self._advanced)
        adv_box = QGroupBox("Advanced")
        form = QFormLayout(adv_box)
        self._device_args = QLineEdit(default_device_args)
        form.addRow("UHD device args:", self._device_args)

        self._rate_msps = QDoubleSpinBox()
        self._rate_msps.setRange(0.25, 40.0)
        self._rate_msps.setDecimals(3)
        self._rate_msps.setValue(4.0)
        self._rate_msps.setSuffix(" Msps")
        form.addRow("Sample rate:", self._rate_msps)

        self._fft = QSpinBox()
        self._fft.setRange(256, 8192)
        self._fft.setSingleStep(256)
        self._fft.setValue(1024)
        form.addRow("FFT size:", self._fft)

        self._peak_max_hold = QCheckBox(
            "Parallel FFT: running max-hold per bin (spectrum above stays real-time / independent)"
        )
        self._peak_max_hold.setChecked(True)
        self._peak_max_hold.setToolTip(
            "If off, peak readout uses each FFT frame only (no accumulation). "
            "Display is always the GNU Radio freq sink."
        )
        form.addRow("Peak bins:", self._peak_max_hold)

        self._peak_stride = QSpinBox()
        self._peak_stride.setRange(1, 128)
        self._peak_stride.setValue(8)
        self._peak_stride.setToolTip(
            "Update peak/SNR text every N FFT vectors (lower = faster refresh, more CPU)."
        )
        form.addRow("Peak readout every N FFTs:", self._peak_stride)

        self._tx_gain = QDoubleSpinBox()
        self._tx_gain.setRange(0.0, 90.0)
        self._tx_gain.setDecimals(1)
        self._tx_gain.setValue(20.0)
        self._tx_gain.setSuffix(" dB")
        form.addRow("TX gain (TX/RX mode):", self._tx_gain)

        self._tx_off_khz = QDoubleSpinBox()
        self._tx_off_khz.setRange(1.0, 5000.0)
        self._tx_off_khz.setDecimals(1)
        self._tx_off_khz.setValue(100.0)
        self._tx_off_khz.setSuffix(" kHz")
        form.addRow("Tone offset / expected peak (from center):", self._tx_off_khz)

        self._tx_ampl = QDoubleSpinBox()
        self._tx_ampl.setRange(0.01, 1.0)
        self._tx_ampl.setDecimals(3)
        self._tx_ampl.setValue(0.2)
        self._tx_ampl.setSingleStep(0.05)
        form.addRow("TX baseband amplitude:", self._tx_ampl)

        adv_layout.addWidget(adv_box)
        self._advanced.setVisible(False)
        root.addWidget(self._advanced)

        self._rb_rx_only.toggled.connect(self._sync_tx_controls)
        self._rb_txrx.toggled.connect(self._sync_tx_controls)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("Start")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        peak_box = QGroupBox("Peak readout")
        peak_layout = QVBoxLayout(peak_box)
        self._peak_view = QLabel()
        self._peak_view.setFont(QFont("monospace", 10))
        self._peak_view.setText("Idle")
        peak_layout.addWidget(self._peak_view)
        root.addWidget(peak_box)

        self._stack = QStackedWidget()
        self._placeholder = QLabel(
            "Expand “Show different mode & receive port…” if you need to change them, then Start.\n"
            "Stop before switching ports (controls lock while running)."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._stack.addWidget(self._placeholder)
        root.addWidget(self._stack, stretch=1)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)

        self._sync_tx_controls()

    def _dispatch_peak(self, text: str) -> None:
        print(text, file=sys.stderr, flush=True)
        self._peak_emitter.text_ready.emit(text)

    def _on_peak_text(self, text: str) -> None:
        self._peak_view.setText(text)

    def _on_advanced_toggled(self, on: bool) -> None:
        self._advanced.setVisible(on)
        self._btn_advanced.setText(
            "Hide advanced parameters…" if on else "Show advanced parameters…"
        )

    def _on_mode_port_toggled(self, on: bool) -> None:
        self._mode_port_panel.setVisible(on)
        self._btn_mode_port.setText(
            "Hide mode & receive port…" if on else "Show different mode & receive port…"
        )

    def _update_mode_port_summary(self) -> None:
        mode = "TX/RX test (tone + RX FFT)" if self._rb_txrx.isChecked() else "RX only"
        if self._rb_a_rx2.isChecked():
            port = "Channel A — RX2"
        elif self._rb_a_txrx.isChecked():
            port = "Channel A — TX/RX"
        elif self._rb_b_rx2.isChecked():
            port = "Channel B — RX2"
        else:
            port = "Channel B — TX/RX"
        self._mode_port_summary.setText(
            f"<b>Current:</b> {mode} · <b>Receive port:</b> {port}"
        )

    def _sync_tx_controls(self) -> None:
        tx_on = self._rb_txrx.isChecked()
        for w in (self._tx_gain, self._tx_ampl):
            w.setEnabled(tx_on)
        # Offset always applies to expected peak frequency (center + offset) for max-hold report.
        self._tx_off_khz.setEnabled(True)

    def _selected_rx_port(self) -> tuple[int, str]:
        """(UHD channel index, RX antenna name)."""
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
        """Lock controls while the flowgraph owns the USRP (prevents unsafe channel switches)."""
        for rb in (
            self._rb_a_rx2,
            self._rb_a_txrx,
            self._rb_b_rx2,
            self._rb_b_txrx,
        ):
            rb.setEnabled(not running)
        self._rb_rx_only.setEnabled(not running)
        self._rb_txrx.setEnabled(not running)
        self._btn_mode_port.setEnabled(not running)
        self._freq_mhz.setEnabled(not running)
        self._rx_gain.setEnabled(not running)
        self._btn_advanced.setEnabled(not running)
        self._advanced.setEnabled(not running)
        for w in (
            self._device_args,
            self._rate_msps,
            self._fft,
            self._peak_max_hold,
            self._peak_stride,
            self._tx_gain,
            self._tx_off_khz,
            self._tx_ampl,
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

    def _on_start(self) -> None:
        restarting = self._tb is not None
        if restarting:
            self._on_stop()
            time.sleep(_USRP_REOPEN_DELAY_S)
            gc.collect()
            QApplication.processEvents()

        cfg = self._build_config()
        try:
            self._tb = B210Flowgraph(cfg, peak_callback=self._dispatch_peak)
        except Exception as e:
            QMessageBox.critical(
                self,
                "UHD / GNU Radio error",
                str(e),
            )
            return

        while self._stack.count() > 1:
            w = self._stack.widget(1)
            self._stack.removeWidget(w)

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
        while self._stack.count() > 1:
            w = self._stack.widget(1)
            w.hide()
            self._stack.removeWidget(w)
        self._tb = None
        gc.collect()
        QApplication.processEvents()
        self._stack.setCurrentIndex(0)
        self.setWindowTitle("USRP B210 — receiver (FFT)")
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_running_state(False)
        self._peak_view.setText("Idle")

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
        help="Default UHD device args (editable under Advanced).",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(default_device_args=args.args)
    win.show()

    def handle_sigint(_s, _f) -> None:
        win.close()

    signal.signal(signal.SIGINT, handle_sigint)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
