from tabulate import tabulate

record = [['model name', 'val_acc', 'filters_conv0', 'filters_conv1', 'n_dense0', 'dropout', 'batch_size', 'epochs', 'notes'],
          ['dogVcats_model.h5', '~0.8', 64, 32, 32, 0.5, 10, 10],
          ['dogVcats_model1.h5', 0.8095, 128, 64, 64, 0.5, 32, 10],
          ['dogVcats_model2.h5', 0.7879, 64, 32, 32, 0.4, 64, 24],
          ['dogVcats_model3.h5', 0.7871, 128, 64, 32, 0.5, 128, 19, "+ filters_conv2=64"]]

print(tabulate(record, headers='firstrow', tablefmt='fancy_grid'))
