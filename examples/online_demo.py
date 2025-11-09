"""Small online training loop demonstrating HTFR usage."""
from __future__ import annotations

import numpy as np

from htfr import HTFRModel, HyperTensor


def make_toy_model() -> HTFRModel:
    ht1 = HyperTensor(
        n=np.array([1.0, 0.0, 0.0]),
        delta=-0.2,
        dneg=-1.0,
        dpos=1.0,
        C=np.array([[0.0, 0.5, 1.0]]),
        tau=0.7,
    )
    ht2 = HyperTensor(
        n=np.array([-0.7, 0.3, 0.6]),
        delta=0.1,
        dneg=-0.8,
        dpos=0.8,
        C=np.array([[1.0, 0.4, -0.1]]),
        tau=0.5,
    )
    return HTFRModel.from_tensors([ht1, ht2], top_k=2)


def target_function(x: np.ndarray) -> float:
    return float(np.sin(x[0]) + 0.25 * x[1] - 0.1 * x[2])


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    model = make_toy_model()
    for step in range(200):
        x = rng.normal(size=3).astype(np.float32)
        y = np.array([target_function(x)], dtype=np.float32)
        yhat = model.predict_and_update(x, y)
        if step % 50 == 0:
            print(
                f"step={step:03d} yhat={float(yhat.item()):+.3f} "
                f"y={float(y.item()):+.3f}"
            )

    x_test = np.array([0.3, -0.2, 0.4], dtype=np.float32)
    prediction = model.predict(x_test)
    print("prediction on x_test:", float(prediction.item()))


if __name__ == "__main__":
    main()
