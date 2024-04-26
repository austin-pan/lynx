import data_loader
from lynx.exp import libfm_bs_exp

rating_table, X_train, y_train, X_test, y_test = data_loader.load_train_test()
libfm_bs_exp.run(
    rating_table,
    X_train,
    y_train,
    X_test,
    y_test,
    seed=data_loader.SEED,
    verbose=data_loader.VERBOSE
)
