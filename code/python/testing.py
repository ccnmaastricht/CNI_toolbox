from cni_toolbox.pRF import pRF
import cni_toolbox.auxiliary as aux
import scipy.io

D = scipy.io.loadmat("/home/mario/temp_data/test_data_pea.mat")

mask = D['mask'].astype(bool)
data = D['wedge_data'][:,mask]

polar_angle = D['polar_angle'][mask]
#eccentricity = D['eccentricity'][mask]
# %%
n_samples, n_rows, n_cols, n_slices = aux.size(data, 4)

parameters = {'f_sampling': 0.666,
              'f_stim': 0.0312,
              'n_samples': n_samples,
              'n_rows': n_rows,
              'n_cols': n_cols,
              'n_slices': n_slices}

# %%
# cd ~/Code/CNI_toolbox/code/python/
from cni_toolbox.PEA import PEA
pea = PEA(parameters)
pea.set_delay(4.0)
pea.set_direction('cw')

# %%
results = pea.fitting(data)

# %%
