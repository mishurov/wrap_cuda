#!/usr/bin/python3

from numpy import genfromtxt
import matplotlib as mpl
import matplotlib.pyplot as plt


plt.style.use("ggplot")
mpl.rcParams["toolbar"] = "None"

#fig = plt.gcf()
fig, ax = plt.subplots(num=None, figsize=(9, 5), dpi=150)
fig.canvas.set_window_title("Wrap CUDA")

plt.xlabel ("Vertices")
plt.ylabel ("Time")

data = genfromtxt(
    "benchmarks.csv",
    delimiter=',',
    names=True,
    dtype=["int","U3","int","int","int"]
)

cpu_data = data[data["Type"] == "CPU"]
gpu_data = data[data["Type"] == "GPU"]

cpu_total, = plt.plot(
    cpu_data["Vertices"], cpu_data["Time"], label="CPU total"
)
gpu_mem, = plt.plot(
    gpu_data["Vertices"], gpu_data["Transfer"], label="GPU memory transfers"
)
gpu_comp, = plt.plot(
    gpu_data["Vertices"], gpu_data["Computation"], label="GPU computations"
)

plt.legend(handles=[cpu_total, gpu_mem, gpu_comp])
plt.title("Wrap deformer statistics")
plt.show()

