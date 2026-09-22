"""与JS共享的分段二次数学规则；不进行本地生产水深求解。"""

from types import SimpleNamespace
import math

STATE_FIELDS = [
    "status",
    "attempt",
    "sweeps",
    "stable",
    "prev_primary",
    "prev_total",
    "base_primary",
    "base_mid",
]


def numeric_ops():
    return SimpleNamespace(
        c=lambda x: x,
        add=lambda a, b: a + b,
        sub=lambda a, b: a - b,
        mul=lambda a, b: a * b,
        div=lambda a, b: a / b,
        min=min,
        max=max,
        abs=abs,
        lt=lambda a, b: a < b,
        lte=lambda a, b: a <= b,
        gt=lambda a, b: a > b,
        eq=lambda a, b: a == b,
        And=lambda a, b: a and b,
        Or=lambda a, b: a or b,
        Not=lambda a: not a,
        choose=lambda t, a, b: a if t else b,
    )


def ee_ops(constant, numbers=False):
    import ee

    return SimpleNamespace(
        c=constant,
        add=lambda a, b: a.add(b),
        sub=lambda a, b: a.subtract(b),
        mul=lambda a, b: a.multiply(b),
        div=lambda a, b: a.divide(b),
        min=lambda a, b: a.min(b),
        max=lambda a, b: a.max(b),
        abs=lambda a: a.abs(),
        lt=lambda a, b: a.lt(b),
        lte=lambda a, b: a.lte(b),
        gt=lambda a, b: a.gt(b),
        eq=lambda a, b: a.eq(b),
        And=lambda a, b: a.And(b),
        Or=lambda a, b: a.Or(b),
        Not=lambda a: a.Not(),
        choose=(
            (lambda t, a, b: ee.Number(ee.Algorithms.If(t, a, b)))
            if numbers
            else lambda t, a, b: b.where(t, a)
        ),
    )


def boundary_terms(wet, dry, slope, peer, cfg, o):
    c, add, sub, mul = o.c, o.add, o.sub, o.mul
    sq = lambda x: mul(x, x)
    down = lambda r: o.div(c(1), add(c(1), sq(r)))
    ws = o.max(c(cfg.sigmaFloor), mul(c(cfg.madScale), wet["mad"]))
    ds = o.max(c(cfg.sigmaFloor), mul(c(cfg.madScale), dry["mad"]))
    variance = add(sq(ws), sq(ds))
    lower = sub(wet["median"], ws)
    upper = add(dry["median"], ds)
    mid = mul(add(lower, upper), c(0.5))
    order = sub(wet["median"], dry["median"])
    ow = o.div(c(1), add(c(1), o.div(sq(o.max(order, c(0))), variance)))
    w = o.min(c(1), o.div(o.min(wet["count"], dry["count"]), c(cfg.minSamplesPerSide)))
    w = mul(
        w,
        mul(
            down(o.div(slope, c(cfg.slopeScaleDeg))),
            mul(
                o.div(
                    c(1), add(c(1), o.div(variance, c(cfg.dispersionScaleMeters**2)))
                ),
                ow,
            ),
        ),
    )
    w = o.choose(o.lte(lower, upper), w, c(0))
    if peer is not None:
        ps = o.max(c(cfg.sigmaFloor), mul(c(cfg.madScale), peer["mad"]))
        excess = o.max(c(0), sub(o.abs(sub(mid, peer["median"])), mul(c(3), ps)))
        w = mul(
            w,
            o.choose(
                o.lte(c(3), peer["count"]), down(o.div(excess, add(ws, ds))), c(1)
            ),
        )
    return dict(lower=lower, upper=upper, mid=mid, weight=w)


def directional_difference(center, plus, minus, dp, dm, hp, hm, o):
    return dict(
        valid=o.Or(hp, hm),
        value=o.choose(
            o.And(hp, hm),
            o.div(o.sub(plus, minus), o.add(dp, dm)),
            o.choose(
                hp, o.div(o.sub(plus, center), dp), o.div(o.sub(center, minus), dm)
            ),
        ),
    )


def has_support(count, xmin, xmax, ymin, ymax, cfg, o):
    return o.And(
        o.lte(o.c(cfg.minAnchors), count),
        o.lte(o.c(cfg.minSpanPixels), o.max(o.sub(xmax, xmin), o.sub(ymax, ymin))),
    )


def coordinate_template(v, o):
    z, big = o.c(0), o.c(1e12)
    regions = []
    for side in (-1, 0, 1):
        for below in (0, 1):
            low = v["upper"] if side == 1 else o.sub(z, big)
            high = v["lower"] if side == -1 else big
            if side == 0:
                low, high = v["lower"], v["upper"]
            low = o.max(low, o.sub(z, big) if below else v["terrain"])
            high = o.min(high, v["terrain"] if below else big)
            low = o.max(low, o.choose(v["hard"], v["terrain"], o.sub(z, big)))
            bw = z if side == 0 else v["boundary"]
            target = v["lower"] if side == -1 else v["upper"]
            tw = v["soft"] if below else z
            regions.append(
                dict(
                    low=low,
                    high=high,
                    feasible=o.lte(low, high),
                    denominator=o.max(
                        o.add(o.add(v["degree"], bw), o.add(tw, v["midWeight"])),
                        o.c(1e-30),
                    ),
                    boundaryNumerator=o.mul(bw, target),
                    otherNumerator=o.add(
                        o.mul(tw, v["terrain"]), o.mul(v["midWeight"], v["mid"])
                    ),
                )
            )
    return dict(v=v, regions=regions, meanDenominator=o.max(v["degree"], o.c(1e-30)))


def minimum_from_template(neighbor_sum, t, o):
    v = t["v"]
    z, best, answer = o.c(0), o.c(1e30), o.c(0)
    sq = lambda x: o.mul(x, x)
    mean = o.div(neighbor_sum, t["meanDenominator"])
    for r in t["regions"]:
        num = o.add(o.add(neighbor_sum, r["boundaryNumerator"]), r["otherNumerator"])
        s = o.max(r["low"], o.min(r["high"], o.div(num, r["denominator"])))
        distance = o.max(z, o.max(o.sub(v["lower"], s), o.sub(s, v["upper"])))
        energy = o.add(
            o.mul(v["degree"], sq(o.sub(s, mean))),
            o.add(
                o.mul(v["boundary"], sq(distance)),
                o.add(
                    o.mul(v["soft"], sq(o.max(z, o.sub(v["terrain"], s)))),
                    o.mul(v["midWeight"], sq(o.sub(s, v["mid"]))),
                ),
            ),
        )
        take = o.And(r["feasible"], o.lt(energy, best))
        answer = o.choose(take, s, answer)
        best = o.choose(take, energy, best)
    return answer


def coordinate_minimum(v, o=None):
    o = o or numeric_ops()
    return minimum_from_template(v["neighborSum"], coordinate_template(v, o), o)


def advance_component(s, r, cfg, o):
    c, a, m, sub = o.c, o.add, o.mul, o.sub
    primary = o.eq(s["status"], c(1))
    secondary = o.eq(s["status"], c(2))
    active = o.Or(primary, secondary)
    rel = lambda x, y: o.div(o.abs(sub(x, y)), o.max(o.abs(y), c(cfg.objectiveFloor)))
    feasible = o.And(o.eq(r["bad"], c(0)), o.lte(r["hard"], c(cfg.hardTolerance)))
    monotone = o.lte(
        r["total"],
        a(
            s["prev_total"],
            m(c(cfg.monotonicTolerance), o.max(c(1), o.abs(s["prev_total"]))),
        ),
    )
    healthy = o.And(feasible, monotone)
    steady = o.And(
        o.lte(rel(r["primary"], s["prev_primary"]), c(cfg.objectiveTolerance)),
        o.lte(rel(r["total"], s["prev_total"]), c(cfg.objectiveTolerance)),
    )
    stable = o.choose(o.And(steady, healthy), a(s["stable"], c(1)), c(0))
    sweeps = a(s["sweeps"], c(cfg.sweepsPerStage))
    converged = o.And(
        healthy,
        o.And(o.lte(r["residual"], c(cfg.residualTolerance)), o.lte(c(2), stable)),
    )
    exhausted = o.lte(c(cfg.maxSweeps), sweeps)
    ready = o.And(primary, converged)
    failed = o.And(primary, o.And(o.Not(converged), o.Or(exhausted, o.Not(healthy))))
    budget = o.max(m(c(cfg.budgetRelative), s["base_primary"]), c(cfg.budgetAbsolute))
    accepted = o.And(
        secondary, o.And(converged, o.lte(sub(r["primary"], s["base_primary"]), budget))
    )
    rejected = o.And(
        secondary,
        o.And(o.Not(accepted), o.Or(converged, o.Or(exhausted, o.Not(healthy)))),
    )
    retry = o.And(rejected, o.lt(a(s["attempt"], c(1)), c(len(cfg.muRatios))))
    fallback = o.And(rejected, o.Not(retry))
    reset = o.Or(ready, retry)
    nxt = dict(
        status=o.choose(
            ready,
            c(2),
            o.choose(
                failed,
                c(5),
                o.choose(accepted, c(3), o.choose(fallback, c(4), s["status"])),
            ),
        ),
        attempt=o.choose(
            ready, c(0), o.choose(retry, a(s["attempt"], c(1)), s["attempt"])
        ),
        sweeps=o.choose(reset, c(0), o.choose(active, sweeps, s["sweeps"])),
        stable=o.choose(reset, c(0), o.choose(active, stable, s["stable"])),
        base_primary=o.choose(ready, r["primary"], s["base_primary"]),
        base_mid=o.choose(ready, r["mid"], s["base_mid"]),
    )
    mu = c(0)
    for i, ratio in enumerate(cfg.muRatios):
        mu = o.choose(o.eq(nxt["attempt"], c(i)), c(cfg.lambdaB * ratio), mu)
    nxt["prev_primary"] = o.choose(
        reset, nxt["base_primary"], o.choose(active, r["primary"], s["prev_primary"])
    )
    nxt["prev_total"] = o.choose(
        reset,
        a(nxt["base_primary"], m(mu, nxt["base_mid"])),
        o.choose(active, r["total"], s["prev_total"]),
    )
    nxt.update(saveBase=ready, restoreBase=o.Or(retry, fallback))
    return nxt
