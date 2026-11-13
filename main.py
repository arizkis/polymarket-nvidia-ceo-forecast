#!/usr/bin/env python3
"""Deterministic empirical quantile forecasts for one make.fun case."""

import bisect
import json
import math
import os
import sys
from pathlib import Path


TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)
HORIZON = 240
BANDS = 8
SHRINKAGE = 7.5
SPREAD_SCALE = 0.5
VOL_FLOOR = 0.00015


def quantile(sorted_values, tau):
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * tau
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(sorted_values):
        return sorted_values[lower]
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[lower + 1] * fraction


def make_band_centers(bands, horizon):
    root_end = math.sqrt(horizon)
    edges = [1.0 + (root_end - 1.0) * i / bands for i in range(bands + 1)]
    centers = []
    for i in range(bands):
        root_center = (edges[i] + edges[i + 1]) * 0.5
        centers.append(max(1, min(horizon, int(root_center * root_center + 0.5))))
    return centers


def clean_prices(raw):
    prices = []
    last = 0.5
    for value in raw if isinstance(raw, list) else ():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = last
        if not math.isfinite(number):
            number = last
        last = min(1.0, max(0.0, number))
        prices.append(last)
    if not prices:
        prices.append(0.5)
    return prices


def market_statistics(prices):
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    if not changes:
        return VOL_FLOOR
    rms = math.sqrt(sum(change * change for change in changes) / len(changes))
    return max(VOL_FLOOR, rms)


def forecast_case(case, bands=BANDS, shrinkage=SHRINKAGE, vol_floor=VOL_FLOOR,
                  spread_scale=SPREAD_SCALE):
    markets = case.get("markets", [])
    horizon = case.get("horizon_minutes", HORIZON)
    if not isinstance(horizon, int) or horizon < 1 or horizon > HORIZON:
        horizon = HORIZON
    if not isinstance(markets, list):
        markets = []

    centers = make_band_centers(bands, horizon)
    pooled = [[] for _ in centers]
    prepared = []

    for market in markets:
        if not isinstance(market, dict):
            market = {}
        prices = clean_prices(market.get("prices"))
        volatility = max(vol_floor, market_statistics(prices))
        band_quantiles = []

        for band_index, step in enumerate(centers):
            deltas = [prices[i + step] - prices[i]
                      for i in range(max(0, len(prices) - step))]
            deltas.sort()
            own = [quantile(deltas, tau) for tau in TAUS]
            band_quantiles.append(own)

            if deltas:
                center = own[2]
                scale = volatility * math.sqrt(step)
                pooled[band_index].extend((change - center) / scale for change in deltas)

        market_id = market.get("id")
        prepared.append((market_id if isinstance(market_id, str) else None,
                         prices, volatility, band_quantiles))

    pooled_quantiles = []
    for values in pooled:
        values.sort()
        pooled_quantiles.append([quantile(values, tau) for tau in TAUS])

    forecasts = []
    root_centers = [math.sqrt(step) for step in centers]
    for market_id, prices, volatility, band_quantiles in prepared:
        current = prices[-1]
        rows = []
        for minute in range(1, horizon + 1):
            root_h = math.sqrt(minute)
            upper = bisect.bisect_right(root_centers, root_h)
            if upper == 0:
                left = right = 0
                fraction = 0.0
            elif upper == len(root_centers):
                left = right = len(root_centers) - 1
                fraction = 0.0
            else:
                left, right = upper - 1, upper
                fraction = ((root_h - root_centers[left]) /
                            (root_centers[right] - root_centers[left]))

            low_step = centers[left]
            high_step = centers[right]
            low_weight = shrinkage / (
                shrinkage + max(1.0, (len(prices) - low_step) / low_step)
            ) if shrinkage > 0 else 0.0
            high_weight = shrinkage / (
                shrinkage + max(1.0, (len(prices) - high_step) / high_step)
            ) if shrinkage > 0 else 0.0
            low_pool = pooled_quantiles[left]
            high_pool = pooled_quantiles[right]
            low_own = band_quantiles[left]
            high_own = band_quantiles[right]

            offsets = []
            for q_index in (0, 1, 3, 4):
                own_low = low_own[q_index] - low_own[2]
                own_high = high_own[q_index] - high_own[2]
                pool_low = ((low_pool[q_index] - low_pool[2]) *
                            volatility * math.sqrt(low_step))
                pool_high = ((high_pool[q_index] - high_pool[2]) *
                             volatility * math.sqrt(high_step))
                low_delta = own_low * (1.0 - low_weight) + pool_low * low_weight
                high_delta = own_high * (1.0 - high_weight) + pool_high * high_weight
                offsets.append(
                    (low_delta * (1.0 - fraction) + high_delta * fraction)
                    * spread_scale
                )

            median = current
            values = [median + offsets[0], median + offsets[1], median,
                      median + offsets[2], median + offsets[3]]
            row = []
            previous = 0.0
            for value in values:
                bounded = min(1.0, max(0.0, value))
                rounded = round(bounded, 6)
                rounded = max(previous, rounded)
                row.append(rounded)
                previous = rounded
            rows.append(row)
        forecasts.append({"id": market_id, "q": rows})
    return {"forecasts": forecasts}


def load_case(input_dir):
    paths = sorted(Path(input_dir).glob("*.in"))
    if not paths:
        return None, None
    path = paths[0]
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source), path.stem
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return None, path.stem


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    case, case_id = load_case(input_dir)
    if case_id is None:
        return 0
    if isinstance(case, dict):
        try:
            result = forecast_case(case)
        except Exception as error:
            print(f"forecast failed; using constant-price fallback: {error}",
                  file=sys.stderr)
            result = {"forecasts": []}
            markets = case.get("markets", [])
            if isinstance(markets, list):
                for market in markets:
                    market_id = market.get("id") if isinstance(market, dict) else None
                    if not isinstance(market_id, str):
                        market_id = None
                    price = clean_prices(market.get("prices") if isinstance(market, dict) else None)[-1]
                    result["forecasts"].append({
                        "id": market_id,
                        "q": [[round(price, 6)] * 5 for _ in range(HORIZON)],
                    })
    else:
        result = {"forecasts": []}

    output_id = case.get("case") if isinstance(case, dict) else case_id
    if (not isinstance(output_id, str) or not output_id or
            not all(character.isascii() and (character.isalnum() or character in "_-")
                    for character in output_id)):
        output_id = case_id
    destination = output_dir / f"{output_id}.out"
    temporary = destination.with_suffix(".tmp")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(result, target, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            target.write("\n")
        temporary.replace(destination)
    except (OSError, TypeError, ValueError) as error:
        print(f"could not write forecast file {destination}: {error}", file=sys.stderr)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    main()
