scatterer = Sphere(n = 1.6, r=.5, center=(5, 5, 5))
medium_index = 1.33
locations = RegularGrid(shape=(200, 200), spacing=.1, normals='z')
wavelen = 0.66

def test_calc_holo():
    holo = calc_holo(scatterer, medium_index, locations, wavelen)

def test_calc_field():
    field = calc_field(scatterer, medium_index, locations, wavelen)

def test_calc_cross_section():
    cross = calc_cross_section(scatterer, medium_index, wavelen)

def test_calc_intensity():
    intensity = calc_intensity(scatterer, medium_index, locations, wavelen)

