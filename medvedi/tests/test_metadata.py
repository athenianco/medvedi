from medvedi import metadata


def test_metadata_smoke():
    assert metadata.__package__ == "medvedi"
    assert isinstance(metadata.__description__, str)
    assert isinstance(metadata.__version__, str)
