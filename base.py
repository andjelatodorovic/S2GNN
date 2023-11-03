import sys
import time
from argparse import Namespace

import torch
import numpy as np
import seaborn as sns

from gcn.utils.learning import train, test
from gcn.utils.tools import get_model, save_results
from gcn.datasets.tools import set_train_val_test_split


def base(data, n, args, seeds):

    print(Namespace(**args))

    n_layers = 3
    cont_repetition = 0

    loss_train_vec = np.zeros((len(seeds), args["epochs"]), )
    loss_val_vec = np.zeros((len(seeds), args["epochs"]), )
    loss_test_vec = np.zeros((len(seeds), args["epochs"]), )
    best_acc_test_vec = np.zeros((len(seeds), args["epochs"]), )
    best_acc_val_vec = np.zeros((len(seeds), args["epochs"]), )
    err_train_vec = np.zeros((len(seeds), args["epochs"]), )
    err_test_vec = np.zeros((len(seeds), args["epochs"]), )
    err_val_vec = np.zeros((len(seeds), args["epochs"]), )

    for seed in seeds:
        print('Executing repetition ' + str(cont_repetition))

        np.random.seed(seed)
        torch.manual_seed(seed)
        if args["device"].type == 'cuda':
            torch.cuda.manual_seed(seed)

        if args["dataset"] in ['Cora', 'Citeseer', 'Pubmed']:
            num_development = 1500
            data = set_train_val_test_split(seed, data, dataset_name=args["dataset"],
                                                num_development=num_development).to(args["device"])
        else:
            num_development = int(0.55*n)
            data = set_train_val_test_split(seed, data, dataset_name=args["dataset"],
                                                num_development=num_development).to(args["device"])

        # getting model
        model = get_model(args, data, n_layers)
        model = model.to(args["device"])

        if args["GraphDifussion"]:
            optimizer = torch.optim.Adam([
                dict(params=model.convs[0].parameters(), weight_decay=args["weight_decay"]),
                {'params': list([p for l in model.convs[1:] for p in l.parameters()]), 'weight_decay': 0}
            ], lr=args["lr"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        best_val_acc = test_acc = 0
        for epoch in range(0, args["epochs"]):
            # print(f'Epoch {epoch+1}/{args["epochs"]}')
                start_time = time.time()
                loss_train_vec[cont_repetition, epoch] = train(model, data, optimizer)
                train_acc, loss_train, val_acc, loss_val, tmp_test_acc, loss_test = test(model, data)
                loss_val_vec[cont_repetition, epoch] = loss_val
                loss_test_vec[cont_repetition, epoch] = loss_test
                err_test_vec[cont_repetition, epoch] = 1 - tmp_test_acc
                err_val_vec[cont_repetition, epoch] = 1 - val_acc
                err_train_vec[cont_repetition, epoch] = 1 - train_acc
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                best_acc_test_vec[cont_repetition, epoch] = test_acc
                best_acc_val_vec[cont_repetition, epoch] = best_val_acc
                end_time = time.time()
                if (args["hyperparameterTunning_mode"] == False) and (epoch == args["epochs"] - 1):
                    log = 'n_layers={:02d}, Epoch={:03d}, Loss train={:.4f}, Loss val={:.4f}, Loss test={:.4f}, ' \
                        'Train acc={:.4f}, Best val acc={:.4f}, Best test acc={:.4f}, Error test={:.4f}, ' \
                        'learning rate={:.6f}, Time={:.4f} seg'
                    print(log.format(n_layers, epoch, loss_train, loss_val, loss_test, train_acc, best_val_acc, test_acc,
                                    err_test_vec[cont_repetition, epoch], optimizer.param_groups[0]['lr'], end_time-start_time))
                sys.stdout.flush()

        results = {
                'hyper': [best_acc_test_vec[:, -1], best_acc_val_vec[:, -1]],
                'train': [loss_train_vec, loss_val_vec, loss_test_vec, best_acc_test_vec, err_test_vec,
                          err_val_vec, err_train_vec]}
        save_results(results, args, n_layers)
        cont_repetition += 1
    if args["verbose"]:
        acc_test_vec_test = best_acc_test_vec[:, -1]
        boots_series = sns.algorithms.bootstrap(acc_test_vec_test, func=np.mean, n_boot=1000)
        test_std_test_seeds = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(acc_test_vec_test)))
        results_log = f'The result for S-SobGNN method in {args["dataset"]} dataset is ' + \
                        f'{np.mean(boots_series)} +- {test_std_test_seeds}'
        print(results_log)
