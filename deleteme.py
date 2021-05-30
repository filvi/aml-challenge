# %%
import numpy as np

a = np.array([])
b = np.array([[2, 2, 2, 2], [1, 1, 1, 1]])
# %%
print(f"prima: {a.shape}")

a = np.vstack((a, b, a, b))
print(f"dopo: {a.shape}")
print(a)
# %%

if np.array([]).shape == (0,):
    print("asd")
# %%
