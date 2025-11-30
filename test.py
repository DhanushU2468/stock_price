import h5py

print("\nInspecting models/lstm_model1.h5 ...")

with h5py.File("models/lstm_model1.h5", "r") as f:
    print("Keys:", list(f.keys()))
    config = f['model_config'][()].decode("utf-8")
    print("\nFirst 500 characters of config:\n")
    print(config[:500])
