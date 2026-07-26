import numpy as np
from astroML.datasets import fetch_sdss_specgals
from Day1AstroData import cKDTree, kdtree_galaxy_nn

galaxies = fetch_sdss_specgals()


ra = np.asarray(galaxies["ra"][:10000])
dec = np.asarray(galaxies["dec"][:10000])

finite = np.isfinite(ra) & np.isfinite(dec)

ra = ra[finite]
dec = dec[finite]
flatcoords = np.column_stack((ra,dec))


def sky_to_unit(ra,dec):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    x = np.cos(dec_rad)*np.cos(ra_rad)
    y = np.cos(dec_rad)*np.sin(ra_rad)
    z=np.sin(dec_rad)
    return np.column_stack((x,y,z))


unit_vectors = sky_to_unit(ra,dec)
norms = np.linalg.norm(unit_vectors, axis=1)
print("All approximately one:", np.allclose(norms, 1.0))

def spherical_kdtgalaxy_nn(ra_deg,dec_deg):
    unit_vectors = sky_to_unit(ra_deg,dec_deg)
    tree = cKDTree(unit_vectors)

    chord_distance, spherical_neighbour = tree.query(unit_vectors, k=2)
    chord_distance = chord_distance[:,1]

    spherical_neighbour = spherical_neighbour[:,1]

    angular_distance_rad = 2*np.arcsin(np.clip(chord_distance / 2, 0 , 1))
    angular_distance_deg = np.rad2deg(angular_distance_rad)
    return spherical_neighbour, angular_distance_deg

sph_neighbours, sph_distance = spherical_kdtgalaxy_nn(ra,dec)
fl_neighbours, fl_distance = kdtree_galaxy_nn(flatcoords)

same_neighbour = (fl_neighbours == sph_neighbours)
changed_percentage = 100 * np.mean(~same_neighbour)

print( f"Neighbour changed for {changed_percentage:.2f}% of galaxies")