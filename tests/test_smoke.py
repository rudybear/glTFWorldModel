import gltfworld
import gltfworld.cli


def test_version():
    assert gltfworld.__version__ == "0.1.0"


def test_cli_main_callable():
    assert callable(gltfworld.cli.main)
