# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:52:07 2025

@author: barrio_o
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fig_to_array(fig):
    """
    Convert a Matplotlib figure to a NumPy RGB array.
    Works with recent Matplotlib versions.
    """
    # Ensure the figure has a canvas
    if not hasattr(fig, 'canvas') or fig.canvas is None:
        FigureCanvas(fig)

    # Remove titles if desired
    for ax in fig.axes:
        ax.set_title("")

    # Draw the figure to the canvas
    fig.canvas.draw()

    # Get width and height
    w, h = fig.canvas.get_width_height()

    # Get RGBA buffer from the renderer
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)  # 4 channels (RGBA)

    # Convert RGBA to RGB by dropping alpha
    buf_rgb = buf[:, :, :3]

    return buf_rgb




def combine_figures(fig_miro, fig_ts, fig_ms, fig_hm):
    """
    Combine multiple Matplotlib figures into one arranged composite layout.
    Layout:
    ┌─────────────────────────────┬─────────────────────────────┐
    │ Temperature and CO, CO₂     │                             │
    │ (fig_miro)                  │                             │
    ├─────────────────────────────┤                             │
    │ Key Ions (fig_ts)           │        Mass Spectra         │
    ├─────────────────────────────┤          (fig_ms)           │
    │ 2D Plot (fig_hm) — bigger   │                             │
    └─────────────────────────────┴─────────────────────────────┘
    """

    # --- Handle tuple inputs and validate ---
    figs = {"fig_miro": fig_miro, "fig_ts": fig_ts, "fig_ms": fig_ms, "fig_hm": fig_hm}
    for name, f in figs.items():
        if isinstance(f, tuple):
            figs[name] = f[0]
        if figs[name] is None:
            raise ValueError(f"{name} is None — please pass a valid Matplotlib figure.")

    # --- Convert all input figures to images ---
    img_miro = fig_to_array(figs["fig_miro"])
    img_ts   = fig_to_array(figs["fig_ts"])
    img_ms   = fig_to_array(figs["fig_ms"])
    img_hm   = fig_to_array(figs["fig_hm"])

    # --- Create new composite figure ---
    fig_combined = plt.figure(figsize=(14, 12), dpi=150)

    # GridSpec for layout
    gs = GridSpec(
        3, 2, figure=fig_combined,
        width_ratios=[1.2, 1],         # left side slightly wider
        height_ratios=[0.7, 0.7, 1.6], # make 2D plot larger
        hspace=0.06,                   # reduce vertical gaps
        wspace=0.02,                   # reduce horizontal gaps
        left=0.03, right=0.98,         # tighter margins
        top=0.96, bottom=0.1
    )

    # --- Left column subplots ---
    ax_miro = fig_combined.add_subplot(gs[0, 0])
    ax_miro.imshow(img_miro, aspect='auto')
    ax_miro.axis("off")
    #ax_miro.set_title("Temperature and gases", fontsize=11, fontweight="bold", pad=1)

    ax_ts = fig_combined.add_subplot(gs[1, 0])
    ax_ts.imshow(img_ts, aspect='auto')
    ax_ts.axis("off")
    #ax_ts.set_title("Key Ions", fontsize=11, fontweight="bold", pad=1)

    ax_hm = fig_combined.add_subplot(gs[2, 0])
    ax_hm.imshow(img_hm, aspect='auto')
    ax_hm.axis("off")
    #ax_hm.set_title("2D Plot", fontsize=11, fontweight="bold", pad=1)

    # --- Right column (mass spectra spans all rows) ---
    ax_ms = fig_combined.add_subplot(gs[:, 1])
    ax_ms.imshow(img_ms, aspect='auto')
    ax_ms.axis("off")
    #ax_ms.set_title("Mass Spectrum", fontsize=11, fontweight="bold", pad=1)

    return fig_combined


