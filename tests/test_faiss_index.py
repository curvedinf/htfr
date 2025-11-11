import numpy as np
import pytest

from htfr.faiss_index import FaissIndex


def test_faiss_index_rebuild_and_search_numpy_fallback() -> None:
    index = FaissIndex(dim=2)
    index._faiss = None  # force NumPy fallback
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    index.rebuild(data)
    dists, indices = index.search(np.array([0.2, 0.1], dtype=np.float16), k=2)
    assert dists.shape == (2,)
    assert indices.shape == (2,)
    assert indices[0] == 0


def test_faiss_index_dimension_mismatch() -> None:
    index = FaissIndex(dim=2)
    with pytest.raises(ValueError):
        index.rebuild(np.ones((4, 3), dtype=np.float16))

    index.rebuild(np.ones((4, 2), dtype=np.float16))
    with pytest.raises(ValueError):
        index.search(np.ones(3, dtype=np.float16), k=1)


def test_faiss_index_handles_empty_data() -> None:
    index = FaissIndex(dim=2)
    index.rebuild(np.zeros((0, 2), dtype=np.float16))
    dists, inds = index.search(np.zeros(2, dtype=np.float16), k=4)
    assert dists.size == 0
    assert inds.size == 0
