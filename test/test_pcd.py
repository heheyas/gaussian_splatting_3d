import pytest

@pytest.fixture
def bin():
    return "./data/person/colmap/sparse/0/points3D.bin"

def test_viz_colmap(bin):
    import open3d as o3d
    from utils.colmap import read_from_colmap
    pts = read_from_colmap(bin)

def test_stats(bin):
    from utils.colmap import stats
    stats(bin)

# if __name__ == "__main__":
#     test_stats(bin())