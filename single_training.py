import torch

from gcn.utils.tools import load_graph_data
from gcn.utils.parser import parse_args
from gcn.datasets.seeds import test_seeds
from gcn.datasets.tools import DATASET_DICT

from base import base



def main(args=None, data_folder='data', output_folder='output'):

    if args is None:
        return

    if not args["no_cuda"]:
        if torch.cuda.device_count() > args["gpu_number"]:
            args["device"] = torch.device('cuda', args["gpu_number"])
            print(f'Setting up GPU {args["gpu_number"]}...')
        else:
            print(f'GPU {args["gpu_number"]} not available, setting up GPU 0...')
            args["device"] = torch.device('cuda', 0)
    else:
        print('GPU not available, setting up CPU...')
        args["device"] = torch.device('cpu')

    name_folder, init_name = DATASET_DICT.get(args["dataset"], [None, None])

    data, n = load_graph_data(args, root_folder=data_folder, name_folder=name_folder, init_name=init_name)

    args['output_folder'] = output_folder

    base(data, n, args, seeds=test_seeds)


if __name__ == '__main__':

    args = parse_args(train_mode=True)
    main(args, data_folder='data', output_folder='single_training')