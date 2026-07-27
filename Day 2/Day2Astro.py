import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_specgals
from scipy import stats
import pandas as pd

data = fetch_sdss_specgals()

# for name in data.dtype.names:
#     print(name)

df = pd.DataFrame(data)
# print(data.head())

df["original_index"] = np.arange(len(df))
#print([col for col in df.columns if "red" in col.lower()]) #Searches through to see if there is any column that has "red" in it

useful = df[
    [
        "original_index",
        "objID",
        "z",
        "modelMag_g",
        "modelMag_r"
    ]
].copy()

useful["g-r"] = useful["modelMag_g"] - useful["modelMag_r"]

#print(useful.describe()) #shows basic stats of the dataset

def summarise_stats(values):
    values = np.asarray(values, dtype = float)

    finite_m = np.isfinite(values)
    finite_values = values[finite_m]

    if len(finite_values) ==0:
        raise ValueError("No finites HERE MATE!")

    #now the functions we want copy from Day2
    mean = np.mean(finite_values)
    median = np.median(finite_values)
    std = np.std(finite_values,ddof = 1)
    variance = np.var(finite_values, ddof = 1)
    q1,q3 = np.percentile(finite_values, [25,75] ) 
    iqr = q3 - q1
    mad = np.median(np.abs(finite_values - median))
    skewness = stats.skew(finite_values, bias = 0)
    kurtosis = stats.kurtosis(finite_values, fisher = True, bias = False)

    return {
        "total count": len(values),
        "finite values": len(finite_values),
        "unfinite_values": len(values) - len(finite_values),
        "mean" : mean,
        "median": median,
        "std": std,
        "variance": variance,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mad": mad,
        "skewness": skewness,
        "kurtosis": kurtosis
    }

variables = {
    "redshift": useful["z"],
    "r magnitude": useful["modelMag_r"],
    "g-r colour": useful["g-r"]
}

atable = pd.DataFrame({
    name: summarise_stats(values)
    for name, values in variables.items()
}).T

print(atable)
 # Compare mean vs median, far apart may mean skewed, long tails,  multiplepopulations
 # If std>>mad then outliers, multimodality, invalid values


def plt_distribution(values, label, bins=50):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    mean = np.mean(values)
    median = np.median(values)

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.7)

    plt.axvline(
        mean,
        linestyle="--",
        label=f"Mean = {mean:.3f}")

    plt.axvline(
        median,
        linestyle=":",
        label=f"Median = {median:.3f}")

    plt.xlabel(label)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

plt_distribution(useful["g-r"], "g-r")
plt_distribution(useful["z"], "redshift")
plt_distribution(useful["modelMag_r"], "magnitude of r band")
plt.show()

def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y

x, y = ecdf(useful["g-r"])

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.xlabel("g-r colour")
plt.ylabel("Cumulative fraction")
plt.tight_layout()
plt.show()

finite_mask = (
    np.isfinite(useful["z"]) &
    np.isfinite(useful["modelMag_g"]) &
    np.isfinite(useful["modelMag_r"]) &
    np.isfinite(useful["g-r"])
)

clean_useful = useful.loc[finite_mask].copy()