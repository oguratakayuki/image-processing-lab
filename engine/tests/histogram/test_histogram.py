import numpy as np

from imglab_engine.histogram.histogram import compute_histogram


def test_all_zero_image():
    image = np.zeros((2, 2), dtype=np.uint8)
    hist = compute_histogram(image)
    assert hist[0] == 4
    assert hist.sum() == 4
    assert np.all(hist[1:] == 0)


def test_two_distinct_values():
    image = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    hist = compute_histogram(image)
    assert hist[0] == 2
    assert hist[255] == 2
    assert hist.sum() == 4


def test_single_pixel():
    image = np.array([[128]], dtype=np.uint8)
    hist = compute_histogram(image)
    assert hist[128] == 1
    assert hist.sum() == 1


def test_multiple_values_with_repeats():
    image = np.array([[10, 20, 30], [10, 10, 50]], dtype=np.uint8)
    hist = compute_histogram(image)
    assert hist[10] == 3
    assert hist[20] == 1
    assert hist[30] == 1
    assert hist[50] == 1
    assert hist.sum() == image.size


def test_output_length_is_256():
    image = np.zeros((3, 3), dtype=np.uint8)
    hist = compute_histogram(image)
    assert hist.shape == (256,)
