import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
y = sys.argv[2]

y_series = df[y]
v_counts = y_series.value_counts(normalize=True)
sorted_v_counts = v_counts.sort_index()
print(np.asarray(sorted_v_counts))