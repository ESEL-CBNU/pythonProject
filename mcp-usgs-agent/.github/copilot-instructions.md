# Copilot Instructions for USGS Forecast MCP Server

## Project Overview

This is an MCP (Model Context Protocol) server for USGS hydrological forecasting. It provides AI agents with tools to fetch stream gauge data from USGS and generate 6-hour ahead water stage (gage height) predictions using a Seq2Seq LSTM neural network.

**Core Purpose**: Multi-step stage forecasting with automated frequency detection and trend visualization.

## Architecture & Data Flow

### Main Components

1. **USGS IV Data Fetcher** (`_fetch_usgs_iv`): 
   - Calls USGS Water Services JSON API (sites + parameter codes 00060/00065)
   - Returns time-indexed DataFrame with discharge (00060) and stage (00065)
   - Handles UTC timestamps and data validation

2. **Frequency Normalizer** (`_infer_and_normalize_freq`):
   - Auto-detects observation interval (5/15/60 minutes) via median diff analysis
   - Reindexes to uniform grid and time-interpolates missing values
   - Critical for consistent model training windows

3. **Seq2Seq LSTM Model** (`Seq2SeqLSTM` class):
   - Encoder: Encodes multivariate input window (both discharge + stage)
   - Decoder: Outputs stage predictions iteratively (teacher forcing during training)
   - Input: N × T_in × F (batch, time window, features)
   - Output: N × T_out (batch, forecast steps)

4. **Training Pipeline** (`_train_or_load_model`):
   - Loads existing model if present (avoid retraining)
   - Falls back to training on 80/20 time-split if new
   - Returns metrics (MAE, RMSE) computed on held-out test set
   - Saves checkpoint to `models/seq2seq_site{site}_f{freq}m_tin{tin}_tout{tout}.pt`

5. **HTML Report Generator** (`_plot_30d_with_forecast`):
   - Plots last 30 days observed data (two Y-axes: stage + discharge)
   - Overlays 6-hour forecast as separate line
   - Generates interactive Plotly HTML with hover, zoom, pan
   - Saves to `reports/usgs_{site}_30d_plus6h_{timestamp}.html`

## Key Workflows

### Data Normalization
The project assumes **stage is the last column** in DataFrames. After fetching USGS IV data:
- Detect frequency automatically (5/15/60 min)
- Resample to uniform grid
- Time-interpolate gaps → forward/backward fill edges
- Normalize values (z-score) per feature

### Model Lifecycle
1. **Load-first pattern**: Always check for existing model before training
2. **Retrain flag**: Pass `retrain=True` to `usgs_stage_forecast_report()` to force retraining
3. **Forecast mode**: Once trained, model runs in eval mode with teacher_forcing=0.0 (no ground truth)
4. **Inverse transform**: Only stage (last feature) is inverse-transformed; users handle discharge separately

### Multi-Step Forecasting
- Input window length (`tin`) scales with `window_hours` and frequency
- Output length (`tout`) scales with `horizon_hours`
- Decoder uses teacher forcing (0.5 ratio) during training, full autoregressive during forecast
- Stage initialized from last observed value; subsequent steps are model predictions

## Project-Specific Conventions

### Naming & Codes
- **00060**: USGS Discharge parameter (streamflow, cubic feet per second)
- **00065**: USGS Stage parameter (gage height, feet)
- **site**: String like "10109000" (8-digit USGS site number)
- **Timestamps**: All UTC unless specified; follow `YYYY-MM-DDTHH:MM:SSZ` format in file names

### Model File Naming
`models/seq2seq_site{site}_f{freq_min}m_tin{tin}_tout{tout}.pt`
- `f{freq_min}m`: Observation frequency (e.g., f5m, f15m, f60m)
- `tin`/`tout`: Input/output sequence lengths in steps

### Report Paths
`reports/usgs_{site}_30d_plus6h_{YYYYMMDDTHHMMSZ}.html`
- Filename encodes site + timestamp of forecast run

### MCP Server Configuration
- **Transport**: stdio (simplest for Cursor/VSCode integration)
- **Entry point**: `.venv/Scripts/python.exe -u mcp_server_usgs_forecast.py`
- **Config file**: `.mcp.json` (keep in sync with actual paths)

## API Tool Signatures

### `usgs_stage_forecast_report(site, start, end, window_hours=24, horizon_hours=6, epochs=10, retrain=False)`
Main tool for forecasting. Returns dict with:
- `forecast_head_rows`: List of dicts (time_utc, stage_pred_00065) for table display
- `html_report_path`: Path to interactive Plotly visualization
- `train_info`: Dict with mae_all_steps, rmse_all_steps, loaded status
- `normalized_freq_minutes`: Detected frequency (5/15/60)

### `usgs_search_sites(state, text="", max_sites=10)`
Lightweight placeholder. Real integration should call USGS waterdata search API.

## Common Pitfalls & Patterns

1. **Missing columns**: Always check that fetched DataFrame has both 00060 and 00065; raise ValueError if not
2. **Frequency detection edge cases**: Min 10 points required; candidates are hard-coded [5, 15, 60] minutes
3. **Last feature assumption**: Scaler and model assume stage is last column; preserve column order after normalization
4. **Model not found**: First run trains a new model (slow); subsequent calls load checkpoint
5. **Torch device**: Hardcoded to CPU ("cpu"); GPU support requires modifying device assignment
6. **Output formatting**: Return `forecast_head_rows` as list of dicts for agent to format into markdown table

## Dependencies & Environment

- **Core**: torch, numpy, pandas
- **USGS Integration**: requests
- **Visualization**: plotly
- **MCP Framework**: mcp[cli]
- **Async**: anyio

Install via: `pip install -r requirements.txt` in virtual environment.

## File Structure

```
.
├── mcp_server_usgs_forecast.py    # Main server code
├── requirements.txt                 # Python dependencies
├── .mcp.json                        # MCP server config
├── .cursor/rules/usgs_forecast.md  # Cursor-specific instructions
├── models/                          # Trained model checkpoints
└── reports/                         # Generated HTML reports
```

## Debugging & Testing

- **Enable verbose MCP logging**: Run server directly (not via stdio) to see print statements in console
- **Test USGS API**: Verify `_fetch_usgs_iv()` with known site (e.g., 10109000) and recent date range
- **Model convergence**: Check epoch-wise train_mse printed to console; target < 0.1 MSE
- **HTML output**: Open report file in browser to verify 30-day trend + forecast line render correctly
