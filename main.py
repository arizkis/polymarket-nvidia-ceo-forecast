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
REGIME_HALF_LIFE = 0.5
SUMREV_KAPPA = 0.0
RECENCY_WEIGHTS = tuple(math.exp2(-age / WEIGHT_HALF_LIFE)
                        for age in range(4320))
MEDIAN_SCALES = {
    "Bill Dally": 1.2,
    "Colette Kress": 0.85,
    "Debora Shoquist": 1.2,
    "Ian Buck": 1.25,
    "Jay Puri": 1.1,
    "Madison Huang": 1.25,
    "Nikesh Arora": 1.2,
    "Chris Malachowsky": 0.85,
    "Jensen Huang (No Change)": 1.15,
    "Johny Srouji": 1.15,
    "Rene Haas": 1.15,
}

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
    cumulative = 0.0
    midpoints = []
    for _, weight in samples:
        cumulative += weight
        midpoints.append((cumulative - 0.5 * weight) / total_weight)
    values = [sample[0] for sample in samples]
    result = []
    for tau in TAUS:
        upper = bisect.bisect_left(midpoints, tau)
        if upper == 0:
            result.append(values[0])
        elif upper == len(values):
            result.append(values[-1])
        else:
            fraction = ((tau - midpoints[upper - 1]) /
                        (midpoints[upper] - midpoints[upper - 1]))
            result.append(values[upper - 1] +
                          fraction * (values[upper] - values[upper - 1]))
    return tuple(result)


def _linear_quantile(sorted_values, tau):
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * tau
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(sorted_values):
        return sorted_values[lower]
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[lower + 1] * fraction


def _local_volatility(prices):
    if len(prices) == 1:
        return [1e-9]
    changes = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    squared_prefix = [0.0]
    for change in changes:
        squared_prefix.append(squared_prefix[-1] + change * change)
    local = [0.0] * len(prices)
    for index in range(1, len(prices)):
        start = max(0, index - 60)
        count = index - start
        local[index] = math.sqrt(
            (squared_prefix[index] - squared_prefix[start]) / count
        )
    local[0] = local[1]
    return local


def _regime_weight(local_vol, vol_now, start):
    start = max(0, start)
    return math.exp(
        -abs(math.log((local_vol[start] + 1e-9) / (vol_now + 1e-9))) /
        REGIME_HALF_LIFE
    )


def _tier(price):
    if price > 0.10:
        return "fav"
    if price >= 0.01:
        return "mid"
    return "long"


def _space_statistics(markets, space):
    pooled_samples = [[] for _ in STEPS]
    pooled_regime_samples = [[] for _ in STEPS]
    market_stats = []
    regime_market_stats = []

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
        regime_own_by_band = []
        local_vol = _local_volatility(prices) if space == "logit" else None
        vol_now = local_vol[-1] if local_vol is not None else None
        regime_factors = (
            [_regime_weight(local_vol, vol_now, index) for index in range(count)]
            if local_vol is not None else None
        )
        for band_index, step in enumerate(STEPS):
            sample_count = count - step
            if sample_count <= 0:
                own_by_band.append((0.0,) * len(TAUS))
                regime_own_by_band.append((0.0,) * len(TAUS))
                continue
            samples = []
            normalizer = vol_rms * math.sqrt(step)
            for index in range(sample_count):
                age = count - 1 - (index + step)
                weight = RECENCY_WEIGHTS[age] if age < len(RECENCY_WEIGHTS) else 0.0
                delta = prices[index + step] - prices[index]
                if local_vol is not None:
                    regime_weight = weight * regime_factors[index]
                else:
                    regime_weight = weight
                samples.append((delta, weight, regime_weight))
            samples.sort(key=lambda sample: sample[0])
            weighted_samples = [(delta, weight) for delta, weight, _ in samples]
            regime_samples = [(delta, weight) for delta, _, weight in samples]
            own = _weighted_quantiles(weighted_samples)
            regime_own = _weighted_quantiles(regime_samples)
            own_by_band.append(own)
            regime_own_by_band.append(regime_own)
            center = own[2]
            regime_center = regime_own[2]
            pooled_samples[band_index].extend(
                ((delta - center) / normalizer, weight)
                for delta, weight, _ in samples
            )
            pooled_regime_samples[band_index].extend(
                ((delta - regime_center) / normalizer, weight)
                for delta, _, weight in samples
            )

        common_stats = {
            "id": market.get("id"),
            "price": checkpoint,
            "count": count,
            "vol_rms": vol_rms,
            "vol_weighted": vol_weighted,
        }
        market_stats.append(dict(common_stats, own=own_by_band))
        regime_market_stats.append(dict(common_stats, own=regime_own_by_band))

    pooled_quantiles = []
    pooled_regime_quantiles = []
    for samples, regime_samples in zip(pooled_samples, pooled_regime_samples):
        samples.sort(key=lambda sample: sample[0])
        regime_samples.sort(key=lambda sample: sample[0])
        pooled_quantiles.append(_weighted_quantiles(samples))
        pooled_regime_quantiles.append(_weighted_quantiles(regime_samples))
    return (market_stats, pooled_quantiles,
            regime_market_stats, pooled_regime_quantiles)


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

    logit_stats = _space_statistics(markets, "logit")
    price_stats = _space_statistics(markets, "price")
    spaces = {
        "logit": (logit_stats[0], logit_stats[1]),
        "logit_regime": (logit_stats[2], logit_stats[3]),
        "price": (price_stats[0], price_stats[1]),
    }

    member_rows = []
    for member_index, member in enumerate(MEMBERS):
        space = "logit_regime" if member_index == 0 else member["space"]
        market_stats, pooled_quantiles = spaces[space]
        member_rows.append(_member_forecasts(
            market_stats, pooled_quantiles, member, horizon
        ))

    raw_prices = [
        _clean_prices(market.get("prices")) if isinstance(market, dict) else [0.5]
        for market in markets
    ]
    checkpoints = [prices[-1] for prices in raw_prices]
    s0 = sum(checkpoints)
    lag_sums = [0.0] * 1440
    for prices in raw_prices:
        for lag in range(1, min(1440, len(prices) - 1) + 1):
            lag_sums[lag - 1] += prices[-1 - lag]
    mean_lag_sum = sum(lag_sums) / len(lag_sums)
    sumrev_base = -SUMREV_KAPPA * (s0 - mean_lag_sum)
    checkpoint_weights = [price * (1.0 - price) for price in checkpoints]
    checkpoint_weight_sum = sum(checkpoint_weights)

    forecasts = []
    for market_index, market in enumerate(markets):
        raw = raw_prices[market_index]
        checkpoint = checkpoints[market_index]
        name = market.get("name") if isinstance(market, dict) else None
        offset_scale = MEDIAN_SCALES.get(name, 1.0)
        sumrev_shift = (
            sumrev_base * checkpoint_weights[market_index] / checkpoint_weight_sum
            if checkpoint_weight_sum else 0.0
        )
        reversion_move = 0.0
        reversion_k = 0.0
        if checkpoint > 0.10 and len(raw) > 90:
            reversion_move = raw[-1] - raw[-91]
            changes_90 = sorted(
                abs(raw[index] - raw[index - 90])
                for index in range(90, len(raw))
            )
            threshold = _linear_quantile(changes_90, 0.80)
            if abs(reversion_move) >= threshold:
                reversion_k = 0.5 if reversion_move > 0.0 else 0.25

        consecutive_differences = (
            abs(raw[index] - raw[index - 1])
            for index in range(1, len(raw))
        )
        tick = min((difference for difference in consecutive_differences
                    if difference > 0.0), default=0.0)

        rows = []
        for minute in range(horizon):
            blended = [
                sum(MEMBERS[member_index]["weight"] *
                    member_rows[member_index][market_index][minute][quantile_index]
                    for member_index in range(len(MEMBERS)))
                for quantile_index in range(len(TAUS))
            ]
            median = blended[2]
            reversion_shift = (
                -reversion_k * reversion_move *
                math.sqrt((minute + 1) / HORIZON)
            )
            shift = reversion_shift + sumrev_shift
            row = []
            previous = 0.0
            for value in blended:
                value = median + shift + (value - median) * offset_scale
                if tick > 0.0:
                    value = round(value / tick) * tick
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
