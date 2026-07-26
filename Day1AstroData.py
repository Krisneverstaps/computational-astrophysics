import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

# gaussian = scipy.stats.norm(loc=1, scale=1)

# uniform = scipy.stats.uniform(loc=-3, scale = 5)

# x = np.linspace(-5,7,100) # generates 100 equally spaced numbers between -5 and 7
# plt.plot(x,gaussian.pdf(x), label = "gaussian")
# plt.plot(x,uniform.pdf(x), label = 'gaussian') 
# plt.show()

# from astroquery.gaia import Gaia

# query = """
# SELECT TOP 20000
#     source_id,
#     ra,
#     dec,
#     pmra,
#     pmdec,
#     parallax,
#     phot_g_mean_mag
# FROM gaiadr3.gaia_source
# WHERE 1 = CONTAINS(
#     POINT('ICRS', ra,dec), 
#     CIRCLE('ICRS', 266.4051, -28.9362, 0.35)
# )
# AND pmra IS NOT NULL
# AND pmdec IS NOT NULL
# AND phot_g_mean_mag IS NOT NULL
# """
# # ICRS is the primary coordinate system used to map star and planet positions
# job = Gaia.launch_job_async(query)
# gaia = job.get_results()

# gaia.write(
#     "data/gaia_crowded_field.csv",
#     format = "ascii.ecsv",
#     overwrite = True,
# )





from astroML.datasets import fetch_sdss_specgals


galaxies = fetch_sdss_specgals()


# JUST BRUSHING UP ON THINGS
# print("Shape:", galaxies.shape)
# print(f"Available columns: {galaxies.dtype.names}")

# print(f"first full galaxy record {galaxies[0]}")

# print("\nfirst 5 redshifts")
# print(galaxies["z"][:5])

# print("\nredshift column shape")
# print( {galaxies["z"].shape}) # there is one redshift measurement for each galaxy

# print(galaxies["ra"])
# print(galaxies[:3])

columns = [
    "ra", # sky coordinates
    "dec",
    "z",
    "petroMag_r", # measure of brightness
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
    "velDisp", # Stellar velocity dispersion
    "d4000", # Strength of 4000 Angstrom break
    "lgm_tot_p50", # Estimated stellar mass
    "sfr_tot_p50", # Estimated SFR

]

def detail_column(data, column_name):
    values = np.asarray(data[column_name])
    finite = np.isfinite(values)
    finite_values = values[finite]

    return {
        "column": column_name,
        "total": finite_values.size,
        "finite?": int(finite.sum()),
        "minimum": finite_values.min(),
        "maximum": finite_values.max(),
        "mean": np.mean(finite_values),
        "median": np.median(finite_values),
        "1, 25, 75, 99 quartile range:": np.percentile(finite_values, [1, 25, 75, 99])

    }


# for column in columns:
#     print(detail_column(galaxies,column))


#Now attempting to apply nearest neighbour searches on this.

X = np.column_stack((galaxies["ra"][:10000], galaxies["dec"][:10000]))


def galaxy_NN(X):
    N, D  = X.shape 
    neighbours = np.zeros(N, dtype = int)
    distances = np.zeros(N, dtype = float)
    for i in range(N):
        j_closest = i
        d_closest = np.inf
        for j in range(N):
            if i ==j:
                continue
            d = np.sqrt(np.sum((X[i] - X[j])**2))
            if d < d_closest:
                d_closest = d
                j_closest = j
        neighbours[i] = j_closest
        distances[i] = d_closest
    return neighbours, distances


def vectorised_galaxy_nn(X):
    XXT = np.dot(X, X.T)
    Xii = XXT.diagonal()

    D = Xii - 2*XXT + Xii[:, np.newaxis]

    neighbours = np.argsort(D, axis = 1)[:,1]
    distances = np.sqrt(D[np.arange(len(X)), neighbours])

    return neighbours, distances

def kdtree_galaxy_nn(X):
    tree = cKDTree(X)

    distances, neighbours = tree.query(X, k=2)
    return neighbours[:, 1], distances[:, 1] # 0 would give its own galaxy


def balltree_galaxy_nn(X):
    tree = BallTree(X)

    distances, neighbours = tree.query(X, k=2)
    return neighbours[:,1], distances[:,1]


# i = range(1000)
# vecneighbours, vecdistances = vectorised_galaxy_nn(X)
# treeneighbours, treedistances = kdtree_galaxy_nn(X)
# ballneighbours, balldistances = balltree_galaxy_nn(X)
# neighbours, distances = galaxy_NN(X)



# comparison = np.column_stack((
#     vecneighbours[i],
#     vecdistances[i],
#     neighbours[i],
#     distances[i],
#     treeneighbours[i],
#     treedistances[i],
#     ballneighbours[i],
#     balldistances[i],
# ))


# for row in comparison:
#     if row[0] != row[2] != row[4] != row[6]:
#         print(f"Mismatch: vectorised neighbour = {int(row[0])}, "
#         f"simple neighbour = {int(row[2])}")


# sample_sizes = [100, 300, 500, 1000, 3000, 5000, 10000]

# all_times = []

# for N in sample_sizes:
#     X = np.column_stack((
#         galaxies["ra"][:N],
#         galaxies["dec"][:N]
#     ))


#     if N <= 3000:

#         start = perf_counter()
#         galaxy_NN(X)
#         time_simple = perf_counter() - start
#     else:
#         time_simple = np.nan

#     start = perf_counter()
#     vectorised_galaxy_nn(X)
#     time_vec = perf_counter() - start

#     start = perf_counter()
#     kdtree_galaxy_nn(X)
#     time_kdt = perf_counter() - start

#     start = perf_counter()
#     balltree_galaxy_nn(X)
#     time_ball = perf_counter() - start

#     all_times.append([
#         N,
#         time_simple,
#         time_vec,
#         time_kdt,
#         time_ball
#     ])

# all_times = np.array(all_times)

# print(all_times)


# plt.plot(all_times[:,0], all_times[:,1], marker = "o", label = "Simple")
# plt.plot(all_times[:,0], all_times[:,2], marker = "o", label = "Vectorise")
# plt.plot(all_times[:,0], all_times[:,3], marker = "o", label = "KDTree")
# plt.plot(all_times[:,0], all_times[:,4], marker = "o", label = "BallTree")

# plt.xscale("log")
# plt.yscale("log")

# plt.xlabel("Number of galaxies")
# plt.ylabel("Runtime/seconds")
# plt.show()


# plt.figure(figsize=(10, 6))

# plt.scatter(
#     galaxies["ra"],
#     galaxies["dec"],
#     s=0.5,
#     alpha=0.5
# )

# plt.xlabel("Right Ascension (deg)")
# plt.ylabel("Declination (deg)")
# plt.title("SDSS Galaxy Sky Positions")
# plt.grid(alpha=0.3)
# plt.show()

#DATA cleaning
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