# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:44:17 2025

@author: barrio_o
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def combined_plotly_ions_gases_left(
    cleaned_df, merged_temp_gases, formulas,
    top_n=5, gases=['CO wet'], mz_min=None, mz_max=None,
    use_log=False, normalize=False, normalize_ions=False, times=None,
    auto_show=True, title=None, save_path=None,
    gas_smoothing="raw",          # "raw", "median", "rolling_median"
    rolling_window=30, tmax=None,
):
    """
    Plotly dashboard: Left column only (Gases + Temperature, Key Ions, Heatmap)
    Normalization:
        - normalize_ions: min-max normalize each ion series
        - normalize_mz: normalize heatmap intensity (mean ± 3*std + min-max)
    """

    # -------------------- Create master subplot layout --------------------
    fig = make_subplots(
        rows=3, cols=1,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
        row_heights=[0.25, 0.25, 0.5],
    )

    # -------------------- GASES + TEMPERATURE --------------------
    df = merged_temp_gases.copy()
    df["t-stamp"] = pd.to_datetime(df["t-stamp"], errors="coerce")
    df = df.dropna(subset=["t-stamp"])
    start_time_abs = df["t-stamp"].iloc[0]
    df["time_s"] = (df["t-stamp"] - start_time_abs).dt.total_seconds()

    if tmax is not None:
        df = df[df["time_s"] <= tmax]

    gas_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    # -------------------- GAS PLOTTING --------------------
    for i, gas in enumerate(gases):
        if gas not in df.columns:
            continue

        color = gas_colors[i % len(gas_colors)]
        y_raw = df[gas] * 1e9

        if normalize and gas_smoothing != "median":
            y_min, y_max = y_raw.min(), y_raw.max()
            if y_max > y_min:
                y_raw = (y_raw - y_min) / (y_max - y_min)

        if gas_smoothing == "raw":
            fig.add_trace(
                go.Scatter(x=df["time_s"], y=y_raw, mode="lines",
                           name=gas, line=dict(width=4, color=color),
                           legendgroup="gases"),
                row=1, col=1, secondary_y=False
            )
        elif gas_smoothing == "median":
            median_val = y_raw.median()
            fig.add_trace(
                go.Scatter(
                    x=[df["time_s"].min(), df["time_s"].max()],
                    y=[median_val, median_val],
                    mode="lines",
                    name=f"{gas} (median)",
                    line=dict(width=4, color=color),
                    legendgroup="gases"),
                row=1, col=1, secondary_y=False
            )
        elif gas_smoothing == "rolling_median":
            y_smooth = y_raw.rolling(rolling_window, center=True).median()
            if normalize:
                s_min, s_max = y_smooth.min(), y_smooth.max()
                if s_max > s_min:
                    y_smooth = (y_smooth - s_min) / (s_max - s_min)
            fig.add_trace(
                go.Scatter(
                    x=df["time_s"], y=y_smooth, mode="lines",
                    name=f"{gas} (rolling median)",
                    line=dict(width=6, color=color),
                    legendgroup="gases"),
                row=1, col=1, secondary_y=False
            )

    # -------------------- TEMPERATURE --------------------
    temp_cols = ["Target Temp (C)", "Measured Temp (C)"]
    temp_colors = ["red", "orange"]
    for i, tcol in enumerate(temp_cols):
        if tcol in df.columns:
            dash_style = "dash" if i == 0 else "solid"
            fig.add_trace(
                go.Scatter(
                    x=df["time_s"], y=df[tcol], mode="lines",
                    name=tcol, line=dict(dash=dash_style, color=temp_colors[i], width=5),
                    legendgroup="temp"),
                row=1, col=1, secondary_y=True
            )

    fig.update_yaxes(title_text="Mixing ratio [ppb]" if not normalize else "Normalized conc.",
                     title_font=dict(size=50), tickfont=dict(size=60),
                     row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temperature [°C]",
                     title_font=dict(size=60), tickfont=dict(size=60),
                     row=1, col=1, secondary_y=True)

    # -------------------- KEY IONS --------------------
    time_col = "time" if "time" in cleaned_df.columns else "Absolute time"
    ion_colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']

    if tmax is not None:
        cleaned_df = cleaned_df[cleaned_df[time_col] <= tmax]

    for i, formula in enumerate(formulas):
        if formula in cleaned_df.columns:
            y_values = cleaned_df[formula].clip(lower=0)
            if normalize_ions:
                y_min, y_max = y_values.min(), y_values.max()
                if y_max > y_min:
                    y_values = (y_values - y_min) / (y_max - y_min)
            fig.add_trace(
                go.Scatter(
                    x=cleaned_df[time_col], y=y_values,
                    mode="lines",
                    name=formula,
                    line=dict(color=ion_colors[i % len(ion_colors)], width=6),
                    legendgroup="ions"
                ),
                row=2, col=1
            )

    fig.update_yaxes(title_text="Normalized intensity" if normalize_ions else "Key ion intensity",
                     title_font=dict(size=60), tickfont=dict(size=60),
                     row=2, col=1)

    # -------------------- HEATMAP --------------------
    mq_cols = [col for col in cleaned_df.columns if col.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in dataframe.")

    mz_data = []
    for col in mq_cols:
        try:
            mz = float(col.split()[1])
            mz_data.append((mz, col))
        except:
            continue

    mz_data.sort(key=lambda x: x[0])
    mz_values = np.array([x[0] for x in mz_data])
    sorted_cols = [x[1] for x in mz_data]

    # Filtering by m/z
    if mz_min is not None or mz_max is not None:
        mz_min_use = mz_min if mz_min is not None else mz_values.min()
        mz_max_use = mz_max if mz_max is not None else mz_values.max()
        mask = (mz_values >= mz_min_use) & (mz_values <= mz_max_use)
        mz_values_filtered = mz_values[mask]
        sorted_cols_filtered = [col for col, keep in zip(sorted_cols, mask) if keep]
    else:
        mz_values_filtered = mz_values
        sorted_cols_filtered = sorted_cols

    heatmap_data = cleaned_df[sorted_cols_filtered].to_numpy().T
    heatmap_data = np.clip(heatmap_data, 1e-12, None)
    
    
    # Auto color scaling , three times standart desviations
    finite_vals = np.ravel(heatmap_data[np.isfinite(heatmap_data)])
    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)

    if use_log:
        heatmap_data = np.log10(heatmap_data)

    else:
        finite_vals = heatmap_data[np.isfinite(heatmap_data)]
        vmin, vmax = 0, mean_val + 3 * std_val

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=cleaned_df[time_col],
            y=mz_values_filtered,
            colorscale="Viridis",
            zmin=vmin, zmax=vmax,
            colorbar=dict(
                title="Intensity" if not use_log else "Intensity (log₁₀)",
                tickfont=dict(size=60),
                len=0.35, thickness=28, x=0.95, y=0.22
            )
        ),
        row=3, col=1
    )

    fig.update_yaxes(title_text="UMR",
                     title_font=dict(size=60), tickfont=dict(size=50),
                     row=3, col=1)

    # -------------------- MS labels --------------------
    if times is None or len(times) == 0:
        times = [25, 620, 780, 985]

    ms_labels = ["MS1", "MS2", "MS3", "MS4"]
    label_y = mz_values_filtered.max() - (mz_values_filtered.max() - mz_values_filtered.min()) * 0.035 * 0.9

    for pos, label in zip(times[:len(ms_labels)], ms_labels):
        fig.add_annotation(
            x=pos, y=label_y, text=label,
            showarrow=False,
            font=dict(size=60, color="white"),
            bgcolor="rgba(0,0,0,0.7)", borderpad=3,
            row=3, col=1
        )

    # -------------------- Layout --------------------
    for r in range(1, 4):
        fig.update_xaxes(title_text="Time [s]", title_font=dict(size=60),
                         tickfont=dict(size=60), showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=False, row=r, col=1)

    fig.update_layout(
        height=3200, width=2400, template="plotly_white",
        title=dict(text=title if title else "", font=dict(size=32), x=0.5, xanchor='center'),
        margin=dict(l=100, r=100, t=100, b=100),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.07,
            xanchor="center", x=0.5,
            font=dict(size=60)
        )
    )

    if save_path is not None:
        fig.write_image(save_path, width=2200, height=2600, scale=2)

    if auto_show:
        fig.show(renderer="browser")

    return fig

def combined_plotly_ions_left(
    cleaned_df, formulas,
    top_n=5, mz_min=None, mz_max=None,
    use_log=False, normalize_ions=False, times=None,
    auto_show=True, title=None, save_path=None, tmax=None,
):
    """
    Plotly dashboard: Left column only (Gases + Temperature, Key Ions, Heatmap)
    """

    # -------------------- Create master subplot layout --------------------
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]],
        row_heights=[0.5, 0.5],
    )

    # -------------------- Subplot Titles --------------------
    fig.add_annotation(
        text="Key Ions + Temperature",
        xref="paper", yref="paper",
        x=0.5, y=1.05, showarrow=False,
        font=dict(size=100)
    )

    fig.add_annotation(
        text="UMR Heatmap",
        xref="paper", yref="paper",
        x=0.5, y=0.47, showarrow=False,
        font=dict(size=100)
    )

    # -------------------- KEY IONS --------------------
    time_col = "time" if "time" in cleaned_df.columns else "Absolute time"
    ion_colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']

    if tmax is not None:
        cleaned_df = cleaned_df[cleaned_df[time_col] <= tmax]

    for i, formula in enumerate(formulas):
        if formula in cleaned_df.columns:
            y_values = cleaned_df[formula].clip(lower=0)
            if normalize_ions:
                y_min, y_max = y_values.min(), y_values.max()
                if y_max > y_min:
                    y_values = (y_values - y_min) / (y_max - y_min)

            fig.add_trace(
                go.Scatter(
                    x=cleaned_df[time_col], y=y_values,
                    mode="lines",
                    name=formula,
                    line=dict(color=ion_colors[i % len(ion_colors)], width=6),
                    legendgroup="ions"
                ),
                row=1, col=1
            )

    # -------------------- TEMPERATURE --------------------
    temp_cols = ["Target Temp (C)", "Measured Temp (C)"]
    temp_colors = ["red", "orange"]

    for i, tcol in enumerate(temp_cols):
        if tcol in cleaned_df.columns:
            
            dash_style = "dash" if i == 0 else "solid"
            fig.add_trace(
                go.Scatter(
                    x=cleaned_df[time_col], y=cleaned_df[tcol], mode="lines",
                    name=tcol, line=dict(dash=dash_style, color=temp_colors[i], width=7),
                    legendgroup="temp"),
                row=1, col=1, secondary_y=True
            )

    fig.update_yaxes(title_text="Intensity",
                     title_font=dict(size=100), tickfont=dict(size=100),
                     row=1, col=1,     secondary_y=False   # LEFT AXIS
)
    fig.update_yaxes(title_text="Temperature [°C]",
                     title_font=dict(size=60), tickfont=dict(size=60),
                     row=1, col=1, secondary_y=True)
    
    fig.update_yaxes(
    title_text="Temperature [°C]",
    title_font=dict(size=100),
    tickfont=dict(size=100),
    row=1,
    col=1,
    secondary_y=True    # RIGHT AXIS
)
    #fig.update_layout(title=dict(text="Key ions and temperature profile [°C] ", font=dict(size=100), x=0.5, xanchor='center'))


    # -------------------- HEATMAP (UMR) --------------------
    mq_cols = [col for col in cleaned_df.columns if col.startswith("m/Q")]
    if not mq_cols:
        raise ValueError("No 'm/Q' columns found in dataframe.")

    mz_data = []
    for col in mq_cols:
        try:
            mz = float(col.split()[1])
            mz_data.append((mz, col))
        except:
            continue

    mz_data.sort(key=lambda x: x[0])
    mz_values = np.array([x[0] for x in mz_data])
    sorted_cols = [x[1] for x in mz_data]

    # Filtering by m/z
    if mz_min is not None or mz_max is not None:
        mz_min_use = mz_min if mz_min is not None else mz_values.min()
        mz_max_use = mz_max if mz_max is not None else mz_values.max()
        mask = (mz_values >= mz_min_use) & (mz_values <= mz_max_use)
        mz_values_filtered = mz_values[mask]
        sorted_cols_filtered = [col for col, keep in zip(sorted_cols, mask) if keep]
    else:
        mz_values_filtered = mz_values
        sorted_cols_filtered = sorted_cols

    heatmap_data = cleaned_df[sorted_cols_filtered].to_numpy().T
    heatmap_data = np.clip(heatmap_data, 1e-12, None)

    # Auto color scaling ±3σ
    finite_vals = np.ravel(heatmap_data[np.isfinite(heatmap_data)])
    mean_val = np.mean(finite_vals)
    std_val = np.std(finite_vals)
    vmax = np.max(finite_vals)

    if use_log:
        heatmap_data = np.log10(heatmap_data)
        vmin, vmax = heatmap_data.min(), heatmap_data.max()
    else:
        vmin, vmax = 0,  mean_val + 3 * std_val
        
        #heatmap_data.max()
       

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=cleaned_df[time_col],
            y=mz_values_filtered,
            colorscale="Viridis",
            zmin=vmin, zmax=vmax,
            colorbar=dict(
                title=dict(
                    text="Intensity" if not use_log else "Intensity (log₁₀)",
                    font=dict(size=100)
                    ),
                tickfont=dict(size=100),
                len=0.35, 
                thickness=40, 
                x=0.95,
                y=0.22
            )
        ),
        row=2, col=1
    )

    fig.update_yaxes(title_text="UMR",
                     title_font=dict(size=100), tickfont=dict(size=100),
                     row=2, col=1)

    # -------------------- MS Labels --------------------
    if times is None or len(times) == 0:
        times = [25, 620, 780, 985]

    ms_labels = ["MS1", "MS2", "MS3", "MS4"]
    label_y = mz_values_filtered.max() - (mz_values_filtered.max() - mz_values_filtered.min()) * 0.035 * 0.9

    for pos, label in zip(times[:len(ms_labels)], ms_labels):
        fig.add_annotation(
            x=pos, y=label_y, text=label,
            showarrow=False,
            font=dict(size=80, color="white"),
            bgcolor="rgba(0,0,0,0.7)", borderpad=5,
            row=2, col=1
        )

    # -------------------- Global Layout & Fonts --------------------
    fig.update_layout(
        height=3500, width=2600, template="plotly_white",
        margin=dict(l=100, r=100, t=150, b=150),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.09,
            xanchor="center", x=0.5,
            font=dict(size=80)
        ),
        font=dict(size=100),
    )

    fig.update_xaxes(title_text="Time [s]", title_font=dict(size=100),
                     tickfont=dict(size=100), showgrid=False,   
                    row=2,col=1
)
    fig.update_yaxes(tickfont=dict(size=100), showgrid=False)

    # -------------------- Save or Show --------------------
    if save_path is not None:
        fig.write_image(save_path, width=2600, height=3500, scale=2)

    if auto_show:
        fig.show(renderer="browser")

    return fig




