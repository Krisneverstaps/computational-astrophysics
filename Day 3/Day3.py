import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import scipy.stats
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
# random_data = pd.DataFrame(["car", "phone", "water"])

# car = pd.Series(["BMW", "Mercedes", "Toyota"])
# colour = pd.Series(["red", "blue", "pink"])
# random_data2 = pd.DataFrame({
#     "car type": car,
#     "colour": colour
# })


# #Importing data

# # pd.read_csv()
# # pd.read_excel()

# #Exporting data

# # colour.to_csv("")



# #Going onto plotting:
# # x = [1,2,3,4]
# # y = [5,6,7,8]

# # fig, ax = plt.subplots() # add an axes (individual plot)
# # ax.plot(x,y)
# # ax.set(title = "Random Sample Plot", xlabel = "x-axis", ylabel = "y-axis")

# # x = np.linspace(1,10,100)
# # fig,ax = plt.subplots()
# # # ax.plot(x,x**2)
# # # ax.scatter(x,np.exp(x))
# # ax.scatter(x, np.sin(x))
# # plt.show()

# different_prices = {
#     "phone": 300,
#     "car": 3000,
#     "envelope":1
# }

# fig,ax = plt.subplots()
# ax.bar(different_prices.keys(), different_prices.values())
# ax.set(title = "Different Prices of Items", xlabel = "Items", ylabel ="prices")
# plt.show()

x = np.linspace(5, -5, 2001)
gaussian_pdf = stats.norm(loc = 0, scale = 1).pdf(x)
laplace_pdf = stats.laplace(loc = 0, scale = 1).pdf(x)
cosine_pdf = stats.cosine(loc = 0, scale = 1).pdf(x)
uniform_pdf = stats.uniform(loc = 2, scale = 4).pdf(x)

# fig, axes = plt.subplots(2,1,figsize=(7, 8),sharex=True)



# axes[0].plot(x, x)

# axes[1].plot(x, laplace_pdf, "--",
#              label=r"Laplace, $K=3$")
# axes[1].plot(x, gaussian_pdf, "-",
#              label=r"Gaussian, $K=0$")
# axes[1].plot(x, cosine_pdf, "-.",
#              label=r"Cosine, $K=-0.59$")
# axes[1].plot(x, uniform_pdf, ":",
#              label=r"Uniform, $K=-1.2$")

# axes[0].set_ylabel(r"$p(x)$")
# axes[1].set_ylabel(r"$p(x)$")
# axes[1].set_xlabel(r"$x$")

# axes[0].legend()
# axes[1].legend()

# plt.tight_layout()
# plt.show()


gaussian = scipy.stats.norm(loc = 1, scale = 1)
cauchy = scipy.stats.cauchy(loc =1, scale = 1)

x = np.linspace(-5,5,1000)

fig, ax = plt.subplots()
ax.plot(x, gaussian.pdf(x), label = "Gaussian")
ax.plot(x, cauchy.pdf(x), label = "Gaussian")



from sklearn.neighbors import KernelDensity
from astropy.visualization import hist
from sklearn.mixture import GaussianMixture


if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True)








plt.rcParams["text.usetex"] = False

# Generate data
random_state = np.random.RandomState(seed=0)
N = 2000

mu_gamma_f = [
    (5, 1.0, 0.1),
    (7, 0.5, 0.5),
    (9, 0.1, 0.1),
    (12, 0.5, 0.2),
    (14, 1.0, 0.1),
]

def hx(values):
    return sum(
        fraction * stats.cauchy(loc=mu, scale=gamma).pdf(values)
        for mu, gamma, fraction in mu_gamma_f
    )

x = np.concatenate([
    stats.cauchy(loc=mu, scale=gamma).rvs(
        size=int(fraction * N),
        random_state=random_state,
    )
    for mu, gamma, fraction in mu_gamma_f
])

random_state.shuffle(x)
x = x[(x > -10) & (x < 30)]

# Plotting grid
fig, ax = plt.subplots(figsize=(10, 10))
xgrid = np.linspace(-10, 30, 1000)

# Histogram
ax.hist(
    x,
    density=True,
    bins=100,
    histtype="step",
    linewidth=2,
    label="Histogram, 100 bins",
)

# Kernel density estimate
kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
kde.fit(x[:, None])

dens_kde = np.exp(
    kde.score_samples(xgrid[:, None])
)

ax.plot(
    xgrid,
    dens_kde,
    label=r"$f(x)$, non-parametric KDE",
)

# Gaussian mixture model
gmm = GaussianMixture(
    n_components=13,
    random_state=0,
)

gmm.fit(x[:, None])

dens_gmm = np.exp(
    gmm.score_samples(xgrid[:, None])
)

ax.plot(
    xgrid,
    dens_gmm,
    label=r"$f(x)$, parametric GMM",
)

# Optional true generating density
# ax.plot(
#     xgrid,
#     hx(xgrid),
#     ":",
#     label=r"$h(x)$, generating distribution",
# )

ax.text(
    0.02,
    0.95,
    f"{len(x)} retained points",
    ha="left",
    va="top",
    transform=ax.transAxes,
)

ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$p(x)$", fontsize=14)
ax.set_xlim(0, 20)
ax.set_ylim(-0.01, 0.4001)
ax.legend(loc="upper right")

plt.show()