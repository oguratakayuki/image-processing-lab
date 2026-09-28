import numpy as np

from imglab_engine.edge.laplacian import laplacian_edge_response


def test_laplacian_responds_strongly_at_isolated_bright_pixel():
    # 周囲が0で中心だけ明るい孤立点。ラプラシアンカーネルは
    # [[0,1,0],[1,-4,1],[0,1,0]]なので、
    #   中心: 0+0+0+0-4*100 = -400 (周囲より大幅に明るい=強い凹み)
    #   上下左右の隣接ピクセル: 100+0+0+0-4*0 = 100 (明るい点が隣にある)
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 100

    response = laplacian_edge_response(image)

    assert response[2, 2] == -400
    assert response[1, 2] == 100
    assert response[3, 2] == 100
    assert response[2, 1] == 100
    assert response[2, 3] == 100


def test_laplacian_is_zero_on_flat_region():
    # 一定値の画像は2階微分が0になるはず(境界の影響を受けない
    # 内側のピクセルで確認)。
    image = np.full((5, 5), 50, dtype=np.uint8)
    response = laplacian_edge_response(image)
    assert response[2, 2] == 0


def test_laplacian_output_is_signed_float():
    image = np.array([[0, 0, 255], [0, 0, 255], [0, 0, 255]], dtype=np.uint8)
    response = laplacian_edge_response(image)
    assert response.dtype == np.float64
    assert (response < 0).any() or (response > 0).any()
