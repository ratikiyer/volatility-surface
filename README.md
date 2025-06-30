# Options Volatility Surface Dashboard

A professional-grade Python Dash application for visualizing, analyzing, and detecting arbitrage in options volatility surfaces. The app fetches real-time options data, computes implied volatilities, and highlights actionable arbitrage opportunities using robust, market-aware logic.

## Features
- **Interactive 3D Volatility Surface**: Visualize implied volatility across strikes and expirations.
- **Real-Time Data Fetching**: Pulls live options chains, spot prices, risk-free rates, and dividend yields from Yahoo Finance.
- **Robust Implied Volatility Calculation**: Handles edge cases, market microstructure, and uses ask/bid for realistic pricing.
- **Arbitrage Detection**: Flags only true, actionable arbitrage (vertical, butterfly, and dominance violations) with no false positives.
- **Modern UI**: Clean, dark-themed dashboard with user-friendly controls and expandable arbitrage alerts.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/options-volatility-surface.git
   cd options-volatility-surface
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Dash app:**
   ```bash
   python app.py
   ```
   The app will be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

2. **Configure and Explore:**
   - Enter a ticker (e.g., `SPY`, `AAPL`) and adjust risk-free rate or other parameters in the configuration bar.
   - View the 3D implied volatility surface and arbitrage alerts.
   - Expand arbitrage cards for trade details.

## Troubleshooting

- **Missing Implied Volatility:**
  - Some options may have missing IV due to illiquid markets, nonsensical prices, or model limitations. This is normal for deep OTM/ITM or stale quotes.
- **Arbitrage Alerts:**
  - Only true, risk-free arbitrage is flagged. If you see unexpected alerts, check your data source and ensure bid/ask prices are valid.
- **Dash Errors:**
  - Ensure you are using a compatible Python version (3.8–3.11 recommended). Some packages may not yet support Python 3.12+.
- **Custom Index Template Errors:**
  - If you modify `app.index_string`, ensure it contains all required Dash placeholders: `{%app_entry%}`, `{%config%}`, `{%scripts%}`.

## Project Structure

```
options-volatility-surface/
├── app.py                # Main Dash app
├── arbitrage.py          # Arbitrage detection logic
├── data_fetch.py         # Data fetching utilities
├── volatility_calc.py    # Implied volatility calculation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes before submitting a PR.

## License
MIT License. See [LICENSE](LICENSE) for details. 