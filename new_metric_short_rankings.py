import numpy as np
import matplotlib.pyplot as plt

def compute_new_metric(distances, pids):

    args = np.argsort(distances)
    pids = pids[args]
    pids_flipped = np.flip(pids)
    ratio = (len(pids)-pids.sum())/pids.sum()
    N = 1
    N_end = 10
    print(N_end)
    quantity = np.zeros(N_end, dtype="float")
    error = np.zeros(N_end, dtype="float")
    index_end = -1
    for i in range(len(quantity)):
        print(N, int(np.ceil(N / ratio)))
        labeled_same = pids[:int(np.ceil(N/ratio))]
        labeled_different = pids_flipped[:N]
        if (N + int(np.ceil(N/ratio)) ) > len(pids):
            index_end = i
            break

        TPS = labeled_same.sum()
        FPS = int(np.ceil(N/ratio)) - labeled_same.sum()
        TPD = N - labeled_different.sum()
        FPD = labeled_different.sum()
        Neutral = len(pids) - N - int(np.ceil(N/ratio))

        quantity[i] = (TPS + TPD) / (TPS + TPD + FPS + FPD + Neutral)
        error[i] =  1. - (TPS + TPD) / (TPS + TPD + FPS + FPD)

        N += 1

    if index_end != -1:
        return quantity[:index_end], error[:index_end]
    else:
        return quantity, error



pids = np.append(np.ones(5),np.zeros(5))



# plot for generating the practival example in chapter:6
distances = np.arange(6)
pids = np.array([0,1,0,1,0,0])

distances = np.arange(10)
pids = np.array([1,1,1,0,1,1,0,0,0,0])
print(pids)
quantity, error = compute_new_metric(distances, pids)

plt.plot(quantity, error, '.-')
#plt.plot(quantity[2], error[2],'ro')
plt.title("The new metric")
plt.xlabel("Quantity")
plt.ylabel("Error")
plt.ylim(-0.05,1)
plt.show()

distances = np.arange(10)
pids = np.array([0,1,1,1,1,1,0,0,0,0])
print(pids)
quantity, error = compute_new_metric(distances, pids)

plt.plot(quantity, error, '.-')
#plt.plot(quantity[2], error[2],'ro')
plt.title("The new metric")
plt.xlabel("Quantity")
plt.ylabel("Error")
plt.ylim(-0.05,1)
plt.show()
