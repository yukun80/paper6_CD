"""v3.2.1云端求解表达式；与任务提交和本地文件管理分离。"""

import math
import ee
from .grid import Grid, DIRS, R, grouped, pack_stage
from .numerics import (
    STATE_FIELDS,
    boundary_terms,
    has_support,
    coordinate_template,
    minimum_from_template,
    advance_component,
    directional_difference,
)


class Solver:
    def __init__(self, components, grid, region, cfg, prepared=None):
        self.g = Grid(grid, region, cfg.p)
        self.cfg = cfg
        self.p = cfg.p
        self.D = components
        self.prepared = prepared
        g = self.g
        self.ids = components.select("cid").updateMask(components.select("support"))
        self.ids_filled = self.ids.unmask(0)
        self.dem = components.select("dem")
        self.support = components.select("support")
        self.hard = components.select("hard")
        self.dry = components.select("dry")
        self.terrain = self.dem.add(self.p.minDepth)
        self.edge = self.hard.Not().And(self.support)
        ref = R * abs(g.t[4]) * math.pi / 180
        self.weights = [
            g.distance(dx, dy)
            .pow(2)
            .pow(-1)
            .multiply(ref * ref * self.p.lambdaC)
            .updateMask(self.ids.eq(g.static_at(self.ids_filled, dx, dy)))
            .unmask(0)
            .updateMask(self.support)
            for dx, dy in DIRS
        ]
        self.degree = g.C(0)
        for weight in self.weights:
            self.degree = self.degree.add(weight)
        self.B = (
            None
            if prepared is None
            else prepared.select(["lower", "upper", "mid", "weight"])
        )
        self.template = None
        self.template_mu = None

    def groups(self, image, names, methods):
        return grouped(image, names, methods, self.ids, self.g)

    def broadcast(self, table, key):
        return (
            self.ids.unmask(0)
            .remap(table.aggregate_array("cid"), table.aggregate_array(key), 0)
            .toDouble()
            .rename(key)
            .setDefaultProjection(self.g.P)
            .updateMask(self.ids.gt(0))
        )

    @staticmethod
    def stats(samples):
        collection = ee.ImageCollection.fromImages(samples)
        median = collection.median().rename("v")
        return {
            "median": median,
            "count": collection.count().unmask(0),
            "mad": ee.ImageCollection.fromImages(
                [im.subtract(median).abs().rename("v") for im in samples]
            ).median(),
        }

    def prepare(self):
        g, p = self.g, self.p
        boundary = self.support.And(
            self.support.unmask(0).focalMin(1, "square", "pixels").Not()
        )
        wet_samples = []
        dry_samples = []
        for dy in range(-p.pairRadius, p.pairRadius + 1):
            for dx in range(-p.pairRadius, p.pairRadius + 1):
                shifted = g.static_at(self.dem, dx, dy).rename("v")
                wet_samples.append(
                    shifted.updateMask(
                        self.ids.eq(g.static_at(self.ids_filled, dx, dy))
                    ).updateMask(boundary)
                )
                touches = g.C(0)
                for ex, ey in DIRS:
                    touches = touches.Or(
                        self.ids.eq(
                            g.static_at(self.ids_filled, dx + ex, dy + ey)
                        ).unmask(0)
                    )
                dry_samples.append(
                    shifted.updateMask(
                        g.static_at(self.dry, dx, dy).And(touches)
                    ).updateMask(boundary)
                )
        wet, dry = self.stats(wet_samples), self.stats(dry_samples)
        slope = ee.Terrain.slope(self.dem)
        first = boundary_terms(wet, dry, slope, None, p, g.ops)
        prelim = first["weight"].updateMask(boundary).unmask(0)
        peers = []
        for py in range(-p.peerRadius, p.peerRadius + 1):
            for px in range(-p.peerRadius, p.peerRadius + 1):
                if px == 0 and py == 0:
                    continue
                peers.append(
                    g.at(first["mid"].updateMask(prelim.gte(p.highWeight)), px, py)
                    .rename("v")
                    .updateMask(self.ids.eq(g.static_at(self.ids_filled, px, py)))
                    .updateMask(boundary)
                )
        peer = self.stats(peers)
        peer = {k: v.unmask(0) if k != "count" else v for k, v in peer.items()}
        weight = (
            boundary_terms(wet, dry, slope, peer, p, g.ops)["weight"]
            .updateMask(boundary)
            .clamp(0, 1)
            .unmask(0)
            .rename("weight")
        )
        self.B = g.pack(
            [first[k].unmask(0).rename(k) for k in ("lower", "upper", "mid")] + [weight]
        ).updateMask(self.support)
        high = weight.gte(p.highWeight)
        xy = g.xy
        stats = g.pack(
            [
                weight.rename("weight_sum"),
                weight.multiply(self.B.select("mid")).rename("weighted_mid"),
                high.rename("anchors"),
                xy.select("x").where(high.Not(), 1e12).rename("xmin"),
                xy.select("x").where(high.Not(), -1e12).rename("xmax"),
                xy.select("y").where(high.Not(), 1e12).rename("ymin"),
                xy.select("y").where(high.Not(), -1e12).rename("ymax"),
            ]
        )

        def eligible(f):
            ok = has_support(
                *[
                    ee.Number(f.get(k))
                    for k in ("anchors", "xmin", "xmax", "ymin", "ymax")
                ],
                p,
                g.nops
            )
            return f.set(
                {
                    "eligible": ok,
                    "initial": ee.Number(f.get("weighted_mid")).divide(
                        ee.Number(f.get("weight_sum")).max(1e-30)
                    ),
                }
            )

        table = self.groups(
            stats,
            ["weight_sum", "weighted_mid", "anchors", "xmin", "xmax", "ymin", "ymax"],
            ["sum", "sum", "sum", "min", "max", "min", "max"],
        ).map(eligible)
        eligibility = self.broadcast(table, "eligible")
        initial = self.broadcast(table, "initial")
        initial = (
            initial.where(self.hard, initial.max(self.terrain))
            .where(eligibility.Not(), 0)
            .rename("S")
        )
        states = {k: g.C(0).updateMask(self.support).rename(k) for k in STATE_FIELDS}
        states["status"] = eligibility.rename("status")
        initial_table = self.groups(
            self.energy(initial, g.C(0)), ["primary", "midterm", "norm"], ["sum"] * 3
        ).map(
            lambda f: f.set(
                "initial_objective",
                ee.Number(f.get("primary")).divide(ee.Number(f.get("norm")).max(1e-30)),
            )
        )
        states["prev_primary"] = self.broadcast(
            initial_table, "initial_objective"
        ).rename("prev_primary")
        states["prev_total"] = states["prev_primary"].rename("prev_total")
        image = (
            self.D.addBands(self.B)
            .addBands(eligibility.rename("eligible"))
            .addBands(initial)
            .addBands(initial.rename("baseS"))
        )
        for key in STATE_FIELDS:
            image = image.addBands(states[key])
        return pack_stage(image, "prepared")

    def mu_for(self, current):
        mu = self.g.C(0)
        status = current.select("status")
        for i, r in enumerate(self.p.muRatios):
            mu = mu.where(
                current.select("attempt").eq(i).And(status.eq(2).Or(status.eq(3))),
                self.p.lambdaB * r,
            )
        return mu

    def neighbor_sum(self, S):
        result = self.g.C(0)
        for (dx, dy), weight in zip(DIRS, self.weights):
            result = result.add(
                self.g.at(S.unmask(0), dx, dy).multiply(weight).unmask(0)
            )
        return result.updateMask(self.support)

    def minimum(self, S, mu):
        if self.template is None or self.template_mu is not mu:
            self.template_mu = mu
            B = self.B
            self.template = coordinate_template(
                {
                    "degree": self.degree,
                    "boundary": B.select("weight").multiply(self.p.lambdaB),
                    "lower": B.select("lower"),
                    "upper": B.select("upper"),
                    "soft": self.edge.multiply(self.p.lambdaT),
                    "terrain": self.terrain,
                    "hard": self.hard,
                    "midWeight": B.select("weight").multiply(mu),
                    "mid": B.select("mid"),
                },
                self.g.ops,
            )
        return (
            minimum_from_template(self.neighbor_sum(S), self.template, self.g.ops)
            .rename("S")
            .updateMask(self.support)
        )

    def energy(self, S, mu):
        continuity = self.g.C(0)
        B = self.B
        p = self.p
        for (dx, dy), weight in zip(DIRS, self.weights):
            continuity = continuity.add(
                S.subtract(self.g.at(S.unmask(0), dx, dy))
                .pow(2)
                .multiply(weight)
                .multiply(0.5)
                .unmask(0)
            )
        distance = (
            B.select("lower").subtract(S).max(S.subtract(B.select("upper"))).max(0)
        )
        primary = (
            continuity.add(
                distance.pow(2).multiply(B.select("weight")).multiply(p.lambdaB)
            )
            .add(
                self.terrain.subtract(S)
                .max(0)
                .pow(2)
                .multiply(self.edge)
                .multiply(p.lambdaT)
            )
            .rename("primary")
        )
        midpoint = (
            S.subtract(B.select("mid"))
            .pow(2)
            .multiply(B.select("weight"))
            .rename("midterm")
        )
        norm = (
            self.degree.multiply(0.5)
            .add(B.select("weight").multiply(p.lambdaB))
            .add(self.edge.multiply(p.lambdaT))
            .rename("norm")
        )
        return self.g.pack(
            [
                primary,
                midpoint,
                norm,
                primary.add(midpoint.multiply(mu)).rename("total"),
            ]
        ).updateMask(self.support)

    def diagnostics(self, S, mu, state):
        g = self.g
        energy = self.energy(S, mu)
        optimal = self.minimum(S, mu)
        bad = (
            g.finite(S)
            .Not()
            .Or(g.finite(optimal).Not())
            .Or(g.finite(energy.select("primary")).Not())
            .Or(g.finite(energy.select("total")).Not())
            .Or(S.mask().unmask(0).Not())
            .unmask(1)
            .And(self.support)
        )
        residual = (
            optimal.subtract(S).abs().where(bad, 1e12).unmask(1e12).rename("residual")
        )
        violation = (
            self.terrain.subtract(S)
            .max(0)
            .multiply(self.hard)
            .where(bad, 1e12)
            .unmask(1e12)
            .rename("hard_violation")
        )
        im = (
            energy.where(bad, 0)
            .unmask(0)
            .addBands(residual)
            .addBands(violation)
            .addBands(bad.rename("bad"))
            .addBands(state.select(STATE_FIELDS))
        )
        names = [
            "primary",
            "total",
            "midterm",
            "norm",
            "residual",
            "hard_violation",
            "bad",
        ] + STATE_FIELDS
        methods = ["sum"] * 4 + ["max", "max", "sum"] + ["min"] * len(STATE_FIELDS)

        def normalize(f):
            norm = ee.Number(f.get("norm")).max(1e-30)
            return f.set(
                {
                    "primary": ee.Number(f.get("primary")).divide(norm),
                    "total": ee.Number(f.get("total")).divide(norm),
                    "mid": ee.Number(f.get("midterm")).divide(norm),
                    "hard": f.get("hard_violation"),
                }
            )

        return self.groups(im, names, methods).map(normalize)

    def sweep(self, S, mu, active):
        for color in range(4):
            S = S.where(
                active.And(self.g.colors.eq(color)), self.minimum(S, mu)
            ).rename("S")
        return S

    def iterate(self, current):
        status = current.select("status")
        mu = self.mu_for(current)
        S = current.select("S")
        active = status.eq(1).Or(status.eq(2))
        for _ in range(self.p.sweepsPerStage):
            S = self.sweep(S, mu, active)

        def advance(f):
            s = {k: ee.Number(f.get(k)) for k in STATE_FIELDS}
            r = {
                k: ee.Number(f.get(k))
                for k in ("primary", "total", "mid", "residual", "hard", "bad")
            }
            return f.set(advance_component(s, r, self.p, self.g.nops))

        table = self.diagnostics(S, mu, current).map(advance)
        base = (
            current.select("baseS")
            .where(self.broadcast(table, "saveBase"), S)
            .rename("baseS")
        )
        S = S.where(self.broadcast(table, "restoreBase"), base).rename("S")
        return pack_stage(
            self.g.pack([S, base] + [self.broadcast(table, k) for k in STATE_FIELDS]),
            "state",
        )

    def gradients(self, surface, valid):
        g = self.g

        def derivative(dx, dy):
            plus = g.at(surface, dx, dy)
            minus = g.at(surface, -dx, -dy)
            r = directional_difference(
                surface,
                plus.unmask(0),
                minus.unmask(0),
                g.distance(dx, dy),
                g.distance(-dx, -dy),
                plus.mask().unmask(0).gt(0),
                minus.mask().unmask(0).gt(0),
                g.ops,
            )
            return r["value"].updateMask(valid.And(r["valid"])).rename("v")

        gx, gy = derivative(1, 0), derivative(0, 1)
        gradient = (
            ee.ImageCollection.fromImages([gx.pow(2), gy.pow(2)])
            .sum()
            .sqrt()
            .updateMask(
                valid.And(gx.mask().unmask(0).gt(0).Or(gy.mask().unmask(0).gt(0)))
            )
            .rename("CFDepth_WSE_Gradient")
            .setDefaultProjection(g.P)
        )
        return gradient

    def finalize(self, current):
        if self.prepared is None:
            raise ValueError("Finalization requires prepared")
        p = self.p
        status = current.select("status")
        S = current.select("S")
        audit = self.diagnostics(S, self.mu_for(current), current)

        def check(f):
            st = ee.Number(f.get("status"))
            accepted = st.eq(3).Or(st.eq(4))
            budget = (
                ee.Number(f.get("base_primary"))
                .multiply(p.budgetRelative)
                .max(p.budgetAbsolute)
            )
            ok = (
                ee.Number(f.get("bad"))
                .eq(0)
                .And(ee.Number(f.get("hard")).lte(p.hardTolerance))
                .And(ee.Number(f.get("residual")).lte(p.residualTolerance))
                .And(
                    ee.Number(f.get("primary"))
                    .subtract(f.get("base_primary"))
                    .lte(budget)
                )
            )
            return f.set("audit_failed", accepted.And(ok.Not()))

        summary = ee.Dictionary(
            {
                "running": audit.filter(ee.Filter.inList("status", [1, 2])).size(),
                "audit_failed": audit.map(check)
                .filter(ee.Filter.eq("audit_failed", 1))
                .size(),
                "failed_components": audit.filter(ee.Filter.eq("status", 5)).size(),
                "unsupported_components": audit.filter(
                    ee.Filter.eq("status", 0)
                ).size(),
            }
        ).getInfo()
        if summary["running"] or summary["audit_failed"]:
            raise ValueError("Final audit rejected: " + str(summary))
        depth = S.subtract(self.dem)
        valid = (
            status.eq(3)
            .Or(status.eq(4))
            .And(self.prepared.select("eligible"))
            .And(self.g.finite(S))
            .And(depth.gt(p.minDepth))
        )
        return {
            "depth": depth.updateMask(valid).rename("CFDepth"),
            "gradient": self.gradients(S.updateMask(valid), valid),
        }, summary
