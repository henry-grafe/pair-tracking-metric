import numpy as np
import matplotlib.pyplot as plt

def generate_simple_distances_list(means, stds,imbalance = 1e-3, valid_set_size = 1000):
    assert len(means) == 2
    assert len(stds) == 2

    distances_valid = np.random.normal(means[0], stds[0], valid_set_size)

    distances_invalid = np.random.normal(means[1], stds[1], int(valid_set_size / imbalance))



    pairs_validity = np.array(np.append(np.ones(len(distances_valid)), np.zeros(len(distances_invalid))), dtype='bool')
    distances_list = np.append(distances_valid, distances_invalid)

    return pairs_validity, distances_list


def generate_distances_list(means, stds, max_overshoot,imbalance = 1e-3, valid_set_size = 1000):
    assert len(means)==2
    assert len(stds)==2

    distances_valid = np.random.normal(means[0], stds[0], valid_set_size)
    inside = np.logical_or(distances_valid>1., distances_valid<0)
    temp_len = len(distances_valid[np.logical_or(inside, np.logical_or(distances_valid>means[0]+stds[0]*max_overshoot[0], distances_valid>1.))])
    while temp_len != 0:
        print(temp_len)
        inside = np.logical_or(distances_valid>1., distances_valid<0)
        distances_valid[np.logical_or(inside, np.logical_or(distances_valid>means[0]+stds[0]*max_overshoot[0], distances_valid>1.))] -= stds[0]*0.2
        temp_len = len(distances_valid[np.logical_or(inside, np.logical_or(distances_valid>means[0]+stds[0]*max_overshoot[0], distances_valid>1.))])

    distances_invalid = np.random.normal(means[1], stds[1], int(valid_set_size/imbalance))
    inside = np.logical_or(distances_invalid>1., distances_invalid<0)
    temp_len = len(distances_invalid[np.logical_or(inside, np.logical_or(distances_invalid<means[1]-stds[1]*max_overshoot[1], distances_invalid<0.))])
    while temp_len != 0:
        print(temp_len)
        inside = np.logical_or(distances_invalid > 1., distances_invalid < 0)
        distances_invalid[np.logical_or(inside, np.logical_or(distances_invalid<means[1]-stds[1]*max_overshoot[1], distances_invalid<0.))] += stds[1]*0.2
        temp_len = len(distances_invalid[np.logical_or(inside, np.logical_or(distances_invalid<means[1]-stds[1]*max_overshoot[1], distances_invalid<0.))])

    pairs_validity = np.array(np.append(np.ones(len(distances_valid)), np.zeros(len(distances_invalid))),dtype='bool')
    distances_list = np.append(distances_valid, distances_invalid)

    args = np.arange(len(distances_list))
    np.random.shuffle(args)

    pairs_validity = pairs_validity[args]
    distances_list = distances_list[args]

    return pairs_validity, distances_list



def generate_distances_list_with_confidence_coefficients(means, stds, max_confidence_stds, imbalance = 1e-3, valid_set_size = 1000):
    assert len(means) == 2
    assert len(stds) == 2
    assert len(max_confidence_stds) == 2

    """
    Valid distances generation, confidence offsets generations and adding them to the distances list
    """
    np.random.seed(42)
    distances_valid = np.random.normal(means[0], stds[0], valid_set_size)

    confidence_coefficients_valid = np.random.random(len(distances_valid))
    confidence_stds_valid = max_confidence_stds[0] * (1-confidence_coefficients_valid)
    confidence_offsets_valid = np.random.normal(0,confidence_stds_valid)

    distances_valid += confidence_offsets_valid


    """
    Invalid distances generation, confidence offsets generations and adding them to the distances list
    """


    distances_invalid = np.random.normal(means[1], stds[1], int(valid_set_size / imbalance))

    confidence_coefficients_invalid = np.random.random(len(distances_invalid))
    confidence_stds_invalid = max_confidence_stds[0] * (1-confidence_coefficients_invalid)
    confidence_offsets_invalid = np.random.normal(0, confidence_stds_invalid)

    distances_invalid += confidence_offsets_invalid

    print(len(distances_valid), len(distances_invalid))

    pairs_validity = np.array(np.append(np.ones(len(distances_valid)), np.zeros(len(distances_invalid))), dtype='bool')
    distances_list = np.append(distances_valid, distances_invalid)
    confidence_coefficients = np.append(confidence_coefficients_valid, confidence_coefficients_invalid)


    args = np.arange(len(distances_list))
    np.random.shuffle(args)

    pairs_validity = pairs_validity[args]
    distances_list = distances_list[args]
    confidence_coefficients = confidence_coefficients[args]

    return pairs_validity, distances_list, confidence_coefficients

"""
pairs_validity, distances_list, confidence_coefficients = generate_distances_list_with_confidence_coefficients([0.4,0.8], [0.1,0.1], [0.1, 0.1], imbalance = 1, valid_set_size = 1000000)

plt.subplot(2,2,1)
plt.title("Distances distribution")
plt.hist(distances_list[pairs_validity], density = True, bins=100, label = "Same identity")
plt.hist(distances_list[np.logical_not(pairs_validity)], density=True, bins=100, alpha=0.7, label = "Different identity")
plt.ylabel("Density")
plt.xlabel("Distance")
plt.legend()
plt.subplot(2,2,2)
plt.title("Confidence coefficients distribution")
plt.hist(confidence_coefficients, bins=100, density=True)
plt.ylabel("Density")
plt.xlabel("Confidence")
plt.show()

args = np.flip(np.argsort(confidence_coefficients))
confidence_coefficients = confidence_coefficients[args]
pairs_validity = pairs_validity[args]
distances_list = distances_list[args]
print(confidence_coefficients)

confs = np.zeros(len(range(0, len(confidence_coefficients), 1000)))
stds = np.zeros(len(range(0, len(confidence_coefficients), 1000)))
c=0
for i in range(0, len(confidence_coefficients), 1000):
    confs[c] = confidence_coefficients[i]
    stds[c] = (distances_list[:i])[pairs_validity[:i]].std()
    c+=1
    print(i)
plt.plot(confs, stds)
plt.title("Standart Deviation evolution as we take in less confident pairs")
plt.xlabel("Minimum confidence coefficient taken")
plt.ylabel("Standart Deviation")
plt.show()
"""