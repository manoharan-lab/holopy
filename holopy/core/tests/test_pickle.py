from holopy.core.tests.common import assert_pickle_roundtrip, get_example_data

def test_image():
    holo = get_example_data('image0001')
    assert_pickle_roundtrip(holo)
