import pytest

_fixture_data = {}

def set_data_for_fixture(key, data):
    _fixture_data[key] = data

@pytest.fixture
def dataset():
    return _fixture_data.get("validation_df")
