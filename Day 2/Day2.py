from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

# dist = stats.uniform(loc=0,scale=2)

# r = dist.rvs(10)

# x = np.linspace(-0.5, 2.5,10000)
# y = dist.pdf(x)

# plt.plot(x,y, c="red")
# plt.title("A random uniform distribution")
# plt.xlabel("Random values")
# plt.ylabel("Set of y values")

# plt.show()

mu, sigma = 0, 0.1
rng = np.random.default_rng()
# s = np.random.normal(mu, sigma, 5000)


# count,bins, ignored = plt.hist(s, 30, density = True)
# plt.plot( bins, 1/(sigma * np.sqrt(2*np.pi)) * np.exp( - (bins - mu)**2 / (2* sigma**2)), linewidth=2,color="r" )


sample1 = rng.normal(mu, sigma, 5000)
sample2 = rng.normal(mu+2, sigma, 10000)
sample3 = rng.normal(mu-2, sigma, 10000)


sample_a = sample1
sample_b = np.concatenate([sample2, sample3])

mean_a = np.mean(sample_a)
mean_b = np.mean(sample_b)

median_a = np.median(sample_a)
median_b = np.median(sample_b)

std_a = np.std(sample_a, ddof = 1)
std_b = np.std(sample_b, ddof = 1)

# plt.hist(sample_a, bins = 50, density = True, alpha = 0.5, label = "Single Gaussian")

# plt.hist(sample_b, bins = 50, density = True, alpha = 0.5, label = "Bimodal distribution") 

plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()

# This shows two distributions with similar means but completely different distributions
#Lets try to make this into a function


def summary_stats(values):
    values = np.asarray(values)

    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values,ddof = 1)
    variance = np.var(values, ddof = 1)
    q1,q3 = np.percentile(values, [25,75] ) 
    iqr = q3 - q1
    mad = np.median(np.abs(values - np.median(values)))
    skewness = stats.skew(values, bias = 0)
    kurtosis = stats.kurtosis(values, fisher = True, bias = False)

    return {
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

#What happens when we have very unclean data? Well we will unclean our data ourselves

clean = sample_a.copy()

unclean_sample = np.concatenate([clean, [-9999], [0.5, 0.7, 1.1]])

print(summary_stats(clean))
print(summary_stats(unclean_sample)) #See how the extremes change our data

plt_sample = unclean_sample[unclean_sample > -10]
plt.hist(clean, bins = 50, density = True, alpha = 0.5, label = "")
plt.hist(plt_sample, bins = 50, density = True, alpha = 0.5, label = "Unclean data")
plt.figure(figsize=(8,5))
plt.show()