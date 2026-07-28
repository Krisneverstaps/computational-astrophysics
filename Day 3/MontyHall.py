import numpy as np
from matplotlib import pyplot as plt


def threedoors(which, n_doors):

    labels = np.arange(n_doors)
    doors = np.zeros(n_doors, dtype =int)
    doors[np.random.choice(labels)] = 1
    choice = np.random.choice(labels)
    notchosen = np.delete(labels, choice)

    opened = []
    
    while len(opened) < n_doors - 2:
        to_open = np.random.choice(notchosen)
        if doors[to_open] == 0 and to_open not in opened:
            opened.append(to_open)

            
    other = int(np.delete(labels, opened + [choice])[0])

           

    if which == "switch":
        return doors[other]

    elif which == "keep":
        return doors[choice]

    elif which == "external":
        picked = np.random.choice([choice, other])
        return doors[picked]


N = int(100000)
n_doors = 10
probs = {}

for which in ["switch", "keep", "external"]:
    events = [threedoors(which, n_doors) for i in range(N)]
    probs[which] = np.sum(events)/ N

plt.bar([0,1,2], [probs[k] for k in probs.keys()], color = "r")
plt.xticks([0,1,2], probs.keys())

for y in [1 / n_doors, 1 / 2, (n_doors - 1) / n_doors]:
    plt.axhline(y, ls=":", c="b")

plt.show()