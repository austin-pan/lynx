import data_loader
from experiments import fastfm_exp

_, X_train, y_train, X_test, y_test = data_loader.load_train_test()
fastfm_exp.run(
    X_train,
    y_train,
    X_test,
    y_test,
    seed=data_loader.SEED
)
