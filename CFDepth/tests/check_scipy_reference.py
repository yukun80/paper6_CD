"""用独立的稀疏图目标与 SciPy 优化器复核生产 JS 坐标求解器。"""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import minimize


def main():
    tests = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="cfdepth_reference_") as temporary:
        fixture = Path(temporary) / "fixture.json"
        subprocess.run(["node", str(tests / "test_numerics.js"), "--dump-fixture", str(fixture)], check=True)
        data = json.loads(fixture.read_text())
    valid = np.flatnonzero(np.asarray(data["labels"]) > 0)
    local = {int(p): i for i, p in enumerate(valid)}
    values = [data["v"][p] for p in valid]
    edges = [(local[p], local[q], w) for p, q, w in data["edges"]]

    def objective(x):
        cost = 0.0
        gradient = np.zeros_like(x)
        for p, q, weight in edges:
            difference = x[p] - x[q]
            cost += weight * difference**2
            gradient[p] += 2 * weight * difference
            gradient[q] -= 2 * weight * difference
        for i, v in enumerate(values):
            distance = min(x[i] - v["lower"], 0) + max(x[i] - v["upper"], 0)
            below = min(x[i] - v["terrain"], 0)
            cost += v["boundary"] * distance**2 + v["soft"] * below**2
            gradient[i] += 2 * v["boundary"] * distance + 2 * v["soft"] * below
        return cost, gradient

    initial = np.array([v["mid"] for v in values])
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 10000, "maxls": 100})
    actual = np.array(data["S"])[valid]
    actual_cost, actual_gradient = objective(actual)
    assert result.success, result.message
    assert abs(actual_cost - result.fun) < 1e-8, (actual_cost, result.fun)
    assert np.max(np.abs(actual_gradient)) < 1e-8
    assert np.max(np.abs(actual - result.x)) < 1e-5
    print(f"Independent SciPy reference passed: F={actual_cost:.12g}, "
          f"objective gap={abs(actual_cost-result.fun):.3g}, "
          f"max WSE difference={np.max(np.abs(actual-result.x)):.3g} m")


if __name__ == "__main__":
    main()
