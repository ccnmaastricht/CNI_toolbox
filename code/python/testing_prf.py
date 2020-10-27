#from cni_toolbox.pRF import pRF
import cni_toolbox.gadgets as gadgets
import scipy.io

D = scipy.io.loadmat("/home/mario/temp_data/test_data_prf.mat")
data = D['data']

#mask = D['mask'].astype(bool)
#data = D['data'][:,mask]

#polar_angle = D['polar_angle'][mask]
eccentricity = D['eccentricity'][mask]
# %%
n_samples, n_rows, n_cols, n_slices = gadgets.size(data, 4)
parameters = {'f_sampling': 0.666,
              'n_samples': n_samples,
              'n_rows': n_rows,
              'n_cols': n_cols,
              'n_slices': n_slices,
              'w_stimulus': 150,
              'h_stimulus': 150}

# %%
from cni_toolbox.pRF import pRF
prf = pRF(parameters)
prf.import_stimulus()

prf.create_timecourses(n_xy = 30, n_slope = 10)
results = prf.mapping(data, mask = mask)

# %%
