import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt
import os
import csv

folder="motif_csv_0.35/"
output_folder = "segmented_signal"

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    motif = np.loadtxt(file_path, delimiter=",")
    Fs = 32000
    N = np.size(motif)
    t=np.arange(N)/Fs

    # Change point detection with L2 cost function and K=3 breakpoints 
    K=4 #Number of breakpoints
    algo = rpt.Dynp(model='rbf', min_size=3).fit(motif)
    my_bkps = algo.predict(n_bkps=K)
    rpt.show.display(motif, my_bkps , my_bkps, figsize=(8, 5))
    plt.show()
    print(my_bkps)


    # start = 0
    # for i, end in enumerate(my_bkps):
    #     segment = motif[start:end]
    #     segment_filename = f"{os.path.splitext(file)[0]}_segment_{i+1}.csv"
    #     output_path = os.path.join(output_folder, segment_filename)

    #     # Save each segment as a CSV file
    #     np.savetxt(output_path, segment, delimiter=",")
    #     print(f"Saved: {output_path}")

    #     start = end  # update for next segment
