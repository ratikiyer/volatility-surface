from __future__ import annotations

import numpy as np
import pandas as pd


def _pct_edge(numerator: float, denominator: float) -> float:
    return abs(numerator) / max(denominator, 1e-9)


def detect_arbitrage(
    calls: pd.DataFrame,
    spot_price: float | None = None,
    r: float = 0.0,
    q: float = 0.0,
    min_edge: float = 0.02,
    min_abs_profit: float = 0.01,
) -> list[str]:
    messages: list[str] = []

    if not {"bid", "ask", "strike", "days_to_expiry"}.issubset(calls.columns):
        return ["No bid/ask data available for robust arbitrage detection."]

    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
    if calls.empty:
        return ["No significant arbitrage opportunities detected."]

    for expiry, grp in calls.groupby("days_to_expiry"):
        grp = grp.sort_values("strike").reset_index(drop=True)
        k = grp["strike"].to_numpy()
        bid = grp["bid"].to_numpy()
        ask = grp["ask"].to_numpy()
        n = len(k)

        for i in range(n - 1):
            credit = bid[i + 1] - ask[i]
            if credit > min_abs_profit and _pct_edge(credit, ask[i]) >= min_edge:
                messages.append(
                    (
                        f"Significant Vertical Dominance Arbitrage at expiry {int(expiry)} days: "
                        f"Buy at ask {k[i]:.2f} (${ask[i]:.2f}), "
                        f"sell at bid {k[i+1]:.2f} (${bid[i+1]:.2f}), "
                        f"credit: ${credit:.2f}, edge: {_pct_edge(credit, ask[i])*100:.1f}%."
                    )
                )

        for i in range(n - 1):
            k1, k2 = k[i], k[i + 1]
            spread = k2 - k1

            cost_to_buy = ask[i] - bid[i + 1]
            if cost_to_buy < -min_abs_profit and _pct_edge(cost_to_buy, ask[i]) >= min_edge:
                messages.append(
                    (
                        f"Significant Call Spread Arbitrage at expiry {int(expiry)} days: "
                        f"Buy at ask {k[i]:.2f} (${ask[i]:.2f}), "
                        f"sell at bid {k[i+1]:.2f} (${bid[i+1]:.2f}), "
                        f"credit: ${-cost_to_buy:.2f} exceeds zero lower bound, "
                        f"edge: {_pct_edge(cost_to_buy, ask[i])*100:.1f}%."
                    )
                )

            credit_to_sell = bid[i] - ask[i + 1]
            if (
                credit_to_sell - spread > min_abs_profit
                and _pct_edge(credit_to_sell - spread, ask[i + 1]) >= min_edge
            ):
                messages.append(
                    (
                        f"Significant Reverse Call Spread Arbitrage at expiry {int(expiry)} days: "
                        f"Sell at bid {k[i]:.2f} (${bid[i]:.2f}), "
                        f"buy at ask {k[i+1]:.2f} (${ask[i+1]:.2f}), "
                        f"net credit: ${credit_to_sell:.2f} exceeds strike diff ${spread:.2f}, "
                        f"edge: {_pct_edge(credit_to_sell - spread, ask[i+1])*100:.1f}%."
                    )
                )

        for i in range(1, n - 1):
            k1, k2, k3 = k[i - 1 : i + 2]
            if not np.isclose(k2 - k1, k3 - k2, atol=1e-8):
                continue
            w1 = (k3 - k2) / (k3 - k1)
            w3 = (k2 - k1) / (k3 - k1)

            lhs = bid[i]
            rhs = w1 * ask[i - 1] + w3 * ask[i + 1]
            excess = lhs - rhs
            if excess > min_abs_profit and _pct_edge(excess, rhs) >= min_edge:
                messages.append(
                    (
                        f"Significant Butterfly Arbitrage at expiry {int(expiry)} days: "
                        f"Buy at ask {k1:.2f} (${ask[i-1]:.2f}) and {k3:.2f} (${ask[i+1]:.2f}), "
                        f"sell at bid {k2:.2f} (${bid[i]:.2f}), "
                        f"net credit: ${excess:.2f}, "
                        f"edge: {_pct_edge(excess, rhs)*100:.1f}%."
                    )
                )

    return messages if messages else ["✅ No significant arbitrage opportunities detected."]
