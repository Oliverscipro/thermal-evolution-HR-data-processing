# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:58:50 2025

@author: barrio_o
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_spectrum_columns(df, excluded_keywords=None):
    """
    Extract numeric m/z columns from dataframe, skipping excluded keywords.
    Returns sorted (cols, mz_values).
    """
    if excluded_keywords is None:
        excluded_keywords = {"target temp", "measured temp",
                             "absolute time", "m/q", "time"}

    spectrum_cols = []
    spectrum_mzs = []

    for col in df.columns:
        col_str = str(col).strip().lower()

        # Skip excluded keywords
        if any(kw in col_str for kw in excluded_keywords):
            continue

        # Must be numeric column
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        # Try interpreting column name as m/z
        try:
            mz_value = float(col)
            spectrum_cols.append(col)
            spectrum_mzs.append(mz_value)
        except ValueError:
            continue

    if not spectrum_cols:
        raise ValueError("No numeric m/z columns found in DataFrame.")

    # Sort by m/z
    sorted_idx = np.argsort(spectrum_mzs)
    spectrum_cols = [spectrum_cols[i] for i in sorted_idx]
    spectrum_mzs = np.array([spectrum_mzs[i] for i in sorted_idx], float)

    return spectrum_cols, spectrum_mzs



def plot_mirror_spec(df_top, df_bottom, times=None, mz_min=100, mz_max=150,
                     figsize=(14, 18), top_n=5, aggregation='mean',
                     formulas_list_top=None, formulas_list_bottom=None,
                     save_path="my_plot.png"):
    """
    Plot mirror mass spectra for top and bottom dataframes.
    Always uses ONLY positive intensities; bottom is mirrored visually.
    """

    df_top = df_top.copy()
    df_bottom = df_bottom.copy()

    # ----------------------------------------------------------
    # 🔥 REMOVE ALL NEGATIVE VALUES FROM BOTH DATAFRAMES
    # ----------------------------------------------------------
    df_top[df_top.select_dtypes(include=[np.number]).columns] = \
        df_top.select_dtypes(include=[np.number]).clip(lower=0)

    df_bottom[df_bottom.select_dtypes(include=[np.number]).columns] = \
        df_bottom.select_dtypes(include=[np.number]).clip(lower=0)
    # ----------------------------------------------------------

    # Auto-select times as single points if not provided
    if times is None:
        if "time" not in df_top.columns:
            raise ValueError("The dataframe must contain a 'time' column.")

        unique_times = np.sort(df_top["time"].unique())
        n_default = 4

        if len(unique_times) < n_default:
            times = [(t, t) for t in unique_times]
        else:
            selected = np.linspace(unique_times.min(),
                                   unique_times.max(),
                                   n_default)
            times = [(t, t) for t in selected]

        print(f"[INFO] Using {len(times)} automatically selected time points.")

    # Ensure list of (start, end)
    processed_times = []
    for t in times:
        if isinstance(t, tuple) and len(t) == 2:
            processed_times.append(t)
        elif isinstance(t, (int, float)):
            processed_times.append((t, t))
        else:
            raise ValueError(f"Invalid time format: {t}.")
    times = processed_times

    # Ensure time numeric
    df_top["time"] = pd.to_numeric(df_top["time"], errors="coerce")
    df_bottom["time"] = pd.to_numeric(df_bottom["time"], errors="coerce")

    # Extract m/z columns
    top_cols, top_mz = extract_spectrum_columns(df_top)
    bot_cols, bot_mz = extract_spectrum_columns(df_bottom)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    global_ymax = 0  # track maximum intensity for y-scaling

    for i, (t_start, t_end) in enumerate(times):

        # ------------------------------------------------------
        # TOP PANEL
        # ------------------------------------------------------
        mask_top = (df_top["time"] >= t_start) & (df_top["time"] <= t_end)
        rows_top = df_top.loc[mask_top]

        if rows_top.empty:
            print(f"Warning: No top data in range ({t_start}, {t_end}).")
            continue

        valid_cols = [c for c in top_cols if c in rows_top]

        if aggregation == "mean":
            intens_top = rows_top[valid_cols].mean().to_numpy()
        elif aggregation == "median":
            intens_top = rows_top[valid_cols].median().to_numpy()
        elif aggregation == "max":
            intens_top = rows_top[valid_cols].max().to_numpy()
        elif aggregation == "sum":
            intens_top = rows_top[valid_cols].sum().to_numpy()

        masses_top = np.array([float(c) for c in valid_cols])

        # Filter m/z range
        mask_mz = (masses_top >= mz_min) & (masses_top <= mz_max)
        intens_top = intens_top[mask_mz]
        masses_top = masses_top[mask_mz]

        # Plot top spectrum
        ax.vlines(masses_top, 0, intens_top, color="#382eaa", lw=10)

        # Label peaks
        if intens_top.size > 0:
            global_ymax = max(global_ymax, intens_top.max())

        if intens_top.size > 0:
            if intens_top.size >= top_n:
                order = np.argsort(intens_top)[-top_n:][::-1]
            else:
                order = np.argsort(intens_top)[::-1]

            labels_top = formulas_list_top[i] if formulas_list_top else None

            for k, idxp in enumerate(order):
                label = labels_top[k] if labels_top else f"{masses_top[idxp]:.1f}"
                ax.text(masses_top[idxp], intens_top[idxp] * 1.10, label,
                        ha="center", va="bottom", fontsize=70, rotation=90)

        # ------------------------------------------------------
        # BOTTOM PANEL (mirrored)
        # ------------------------------------------------------
        mask_bot = (df_bottom["time"] >= t_start) & (df_bottom["time"] <= t_end)
        rows_bot = df_bottom.loc[mask_bot]

        if rows_bot.empty:
            print(f"Warning: No bottom data in range ({t_start}, {t_end}).")
            continue

        valid_cols_b = [c for c in bot_cols if c in rows_bot]

        if aggregation == "mean":
            intens_bot = rows_bot[valid_cols_b].mean().to_numpy()
        elif aggregation == "median":
            intens_bot = rows_bot[valid_cols_b].median().to_numpy()
        elif aggregation == "max":
            intens_bot = rows_bot[valid_cols_b].max().to_numpy()
        elif aggregation == "sum":
            intens_bot = rows_bot[valid_cols_b].sum().to_numpy()

        masses_bot = np.array([float(c) for c in valid_cols_b])

        # Filter m/z
        mask_mzb = (masses_bot >= mz_min) & (masses_bot <= mz_max)
        intens_bot = intens_bot[mask_mzb]
        masses_bot = masses_bot[mask_mzb]

        # Mirror plot
        ax.vlines(masses_bot, -intens_bot, 0, color="#aa2e2e", lw=10)

        if intens_bot.size > 0:
            global_ymax = max(global_ymax, intens_bot.max())

        # Label peaks
        if intens_bot.size > 0:
            if intens_bot.size >= top_n:
                orderb = np.argsort(intens_bot)[-top_n:][::-1] #to get the correct order in the annotations
            else:
                orderb = np.argsort(intens_bot)

            labels_bot = formulas_list_bottom[i] if formulas_list_bottom else None

            for k, idxp in enumerate(orderb):
                label = labels_bot[k] if labels_bot else f"{masses_bot[idxp]:.1f}"
                ax.text(masses_bot[idxp], -intens_bot[idxp] * 1.05, label,
                        ha="center", va="top", fontsize=70, rotation=90)

    # ------------------------------------------------------
    # FINAL PLOT STYLING
    # ------------------------------------------------------
    ax.set_xlabel("m/z", fontsize=100, fontweight='bold')
    ax.set_ylabel("Intensity", fontsize=100, fontweight="bold")
    ax.set_xlim(mz_min, mz_max)

    ax.set_ylim(-1.4 * global_ymax, 1.4 * global_ymax)

    ax.tick_params(axis='both', labelsize=100)
    ax.set_title("MS4", fontsize=100, loc="right")
    ax.axhline(y=0, color='black', linewidth=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Figure saved to: {save_path}")

    return fig, ax