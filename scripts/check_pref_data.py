import sys
import time

from src.pref_data import load_hh, load_uf

which, n_train, n_val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fn, tr, va = {"hh": (load_hh, "train", "test"),
              "uf": (load_uf, "train_prefs", "test_prefs")}[which]

t = time.time()
train = fn(tr, n_train, 42)
print(f"[{which}] train: {len(train)} of {n_train} wanted ({time.time() - t:.0f}s)", flush=True)
t = time.time()
val = fn(va, n_val, 42)
print(f"[{which}] val:   {len(val)} of {n_val} wanted ({time.time() - t:.0f}s)", flush=True)

ex = train[0]
for k in ("prompt", "chosen", "rejected"):
    print(f"{k.upper():9}:", ex[k][:200].replace("\n", " "), flush=True)
