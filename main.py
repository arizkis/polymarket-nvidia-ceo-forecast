#!/usr/bin/env python3
"""Deterministic three-member empirical quantile ensemble."""

import bisect
import json
import math
import os
import sys
from pathlib import Path


TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)
OFFSETS = (0, 1, 3, 4)
HORIZON = 240
STEPS = (4, 14, 31, 54, 84, 120, 163, 213)
SQRT_STEPS = tuple(math.sqrt(step) for step in STEPS)
VOL_FLOOR = 0.00015
WEIGHT_HALF_LIFE = 720.0
RECENCY_WEIGHTS = tuple(math.exp2(-age / WEIGHT_HALF_LIFE)
                        for age in range(4320))

MEMBERS = (
    {
        "weight": 0.6,
        "space": "logit",
        "shrink": 50.0,
        "scale": 0.575,
        "band_shrink": (1.0,) * 8,
        "band_scale": (1.0,) * 8,
        "long_scale": 1.2,
        "vol": "weighted",
    },
    {
        "weight": 0.2,
        "space": "logit",
        "shrink": 15.0,
        "scale": 0.5,
        "band_shrink": (0.5, 2.25, 2.25, 1.0, 2.25, 2.25, 2.25, 2.25),
        "band_scale": (1.15, 1.15, 1.15, 1.15, 1.075, 1.0, 1.15, 1.15),
        "long_scale": 1.2,
        "vol": "weighted",
    },
    {
        "weight": 0.2,
        "space": "price",
        "shrink": 7.5,
        "scale": 0.5,
        "band_shrink": (1.0,) * 8,
        "band_scale": (1.0,) * 8,
        "long_scale": 1.0,
        "vol": "rms",
    },
)


def _logit(price):
    clipped = min(0.9995, max(0.0005, price))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value):
    if value >= 0.0:
        e = math.exp(-value)
        return 1.0 / (1.0 + e)
    e = math.exp(value)
    return e / (1.0 + e)


def _clean_prices(raw):
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


def _weighted_quantiles(samples):
    """Midpoint-CDF weighted quantiles; samples are sorted (value, weight)."""
    if not samples:
        return (0.0,) * len(TAUS)
    total_weight = sum(sample[1] for sample in samples)
    first_u = 0.5 * samples[0][1] / total_weight
    last_u = 1.0 - 0.5 * samples[-1][1] / total_weight
    result = []
    for tau in TAUS:
        if tau <= first_u:
            result.append(samples[0][0])
            continue
        if tau >= last_u:
            result.append(samples[-1][0])
            continue
        cumulative = 0.0
        previous_value = samples[0][0]
        previous_u = first_u
        answer = samples[-1][0]
        for value, weight in samples:
            cumulative += weight
            current_u = (cumulative - 0.5 * weight) / total_weight
            if current_u >= tau:
                fraction = (tau - previous_u) / (current_u - previous_u)
                answer = previous_value + fraction * (value - previous_value)
                break
            previous_value = value
            previous_u = current_u
        result.append(answer)
    return tuple(result)


def _tier(price):
    if price > 0.10:
        return "fav"
    if price >= 0.01:
        return "mid"
    return "long"


def _space_statistics(markets, space):
    pooled_samples = [[] for _ in STEPS]
    market_stats = []

    for market in markets:
        raw_prices = _clean_prices(market.get("prices"))
        checkpoint = raw_prices[-1]
        prices = ([_logit(value) for value in raw_prices]
                  if space == "logit" else raw_prices)
        count = len(prices)
        changes = [prices[i] - prices[i - 1] for i in range(1, count)]
        if changes:
            vol_rms = max(
                VOL_FLOOR,
                math.sqrt(sum(change * change for change in changes) / len(changes)),
            )
            weighted_sum = 0.0
            weight_sum = 0.0
            for age, change in enumerate(reversed(changes)):
                weight = RECENCY_WEIGHTS[age] if age < len(RECENCY_WEIGHTS) else 0.0
                weighted_sum += weight * change * change
                weight_sum += weight
            vol_weighted = max(
                VOL_FLOOR,
                math.sqrt(weighted_sum / weight_sum) if weight_sum else 0.0,
            )
        else:
            vol_rms = VOL_FLOOR
            vol_weighted = VOL_FLOOR

        own_by_band = []
        for band_index, step in enumerate(STEPS):
            sample_count = count - step
            if sample_count <= 0:
                own_by_band.append((0.0,) * len(TAUS))
                continue
            samples = []
            normalizer = vol_rms * math.sqrt(step)
            for index in range(sample_count):
                age = count - 1 - (index + step)
                weight = RECENCY_WEIGHTS[age] if age < len(RECENCY_WEIGHTS) else 0.0
                delta = prices[index + step] - prices[index]
                samples.append((delta, weight))
            samples.sort()
            own = _weighted_quantiles(samples)
            own_by_band.append(own)
            center = own[2]
            pooled_samples[band_index].extend(
                ((delta - center) / normalizer, weight)
                for delta, weight in samples
            )

        market_stats.append({
            "id": market.get("id"),
            "price": checkpoint,
            "count": count,
            "vol_rms": vol_rms,
            "vol_weighted": vol_weighted,
            "own": own_by_band,
        })

    pooled_quantiles = []
    for samples in pooled_samples:
        samples.sort()
        pooled_quantiles.append(_weighted_quantiles(samples))
    return market_stats, pooled_quantiles


def _member_forecasts(market_stats, pooled_quantiles, member, horizon):
    all_rows = []
    root_horizons = [math.sqrt(hour) for hour in range(1, horizon + 1)]
    for stats in market_stats:
        checkpoint = stats["price"]
        tier = _tier(checkpoint)
        base = _logit(checkpoint) if member["space"] == "logit" else checkpoint
        vol_rescale = stats["vol_weighted"] if member["vol"] == "weighted" else stats["vol_rms"]
        tier_scale = member["long_scale"] if tier == "long" else 1.0
        band_offsets = []

        for band_index, step in enumerate(STEPS):
            own = stats["own"][band_index]
            pool = pooled_quantiles[band_index]
            n_eff = max(1.0, (stats["count"] - step) / step)
            shrink = member["shrink"] * member["band_shrink"][band_index]
            pool_weight = shrink / (shrink + n_eff)
            offsets = []
            for quantile_index in OFFSETS:
                own_offset = own[quantile_index] - own[2]
                pool_offset = ((pool[quantile_index] - pool[2]) *
                               vol_rescale * math.sqrt(step))
                offsets.append(
                    (own_offset * (1.0 - pool_weight) +
                     pool_offset * pool_weight) *
                    member["band_scale"][band_index]
                )
            band_offsets.append(offsets)

        rows = []
        for sqrt_h in root_horizons:
            upper = bisect.bisect_right(SQRT_STEPS, sqrt_h)
            if upper == 0:
                left = right = 0
                fraction = 0.0
            elif upper == len(STEPS):
                left = right = len(STEPS) - 1
                fraction = 0.0
            else:
                left, right = upper - 1, upper
                fraction = ((sqrt_h - SQRT_STEPS[left]) /
                            (SQRT_STEPS[right] - SQRT_STEPS[left]))
            factor = member["scale"] * tier_scale
            offsets = [
                (band_offsets[left][index] * (1.0 - fraction) +
                 band_offsets[right][index] * fraction) * factor
                for index in range(4)
            ]
            row = [base + offsets[0], base + offsets[1], base,
                   base + offsets[2], base + offsets[3]]
            if member["space"] == "logit":
                row = [_sigmoid(value) for value in row]
            row = [min(1.0, max(0.0, value)) for value in row]
            rows.append(row)
        all_rows.append(rows)
    return all_rows


def forecast_case(case):
    markets = case.get("markets", [])
    if not isinstance(markets, list):
        markets = []
    horizon = case.get("horizon_minutes", HORIZON)
    if not isinstance(horizon, int) or horizon < 1 or horizon > HORIZON:
        horizon = HORIZON

    spaces = {}
    for space in ("logit", "price"):
        spaces[space] = _space_statistics(markets, space)

    member_rows = []
    for member in MEMBERS:
        market_stats, pooled_quantiles = spaces[member["space"]]
        member_rows.append(_member_forecasts(
            market_stats, pooled_quantiles, member, horizon
        ))

    forecasts = []
    for market_index, market in enumerate(markets):
        rows = []
        for minute in range(horizon):
            blended = [
                sum(MEMBERS[member_index]["weight"] *
                    member_rows[member_index][market_index][minute][quantile_index]
                    for member_index in range(len(MEMBERS)))
                for quantile_index in range(len(TAUS))
            ]
            row = []
            previous = 0.0
            for value in blended:
                rounded = round(min(1.0, max(0.0, value)), 6)
                rounded = max(previous, rounded)
                row.append(rounded)
                previous = rounded
            rows.append(row)
        market_id = market.get("id") if isinstance(market, dict) else None
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
                    price = _clean_prices(
                        market.get("prices") if isinstance(market, dict) else None
                    )[-1]
                    result["forecasts"].append({
                        "id": market_id,
                        "q": [[round(price, 6)] * 5 for _ in range(HORIZON)],
                    })
    else:
        result = {"forecasts": []}

    output_id = case.get("case") if isinstance(case, dict) else case_id
    if (not isinstance(output_id, str) or not output_id or
            not all(character.isascii() and
                    (character.isalnum() or character in "_-")
                    for character in output_id)):
        output_id = case_id
    destination = output_dir / f"{output_id}.out"
    temporary = destination.with_suffix(".tmp")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(result, target, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
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
