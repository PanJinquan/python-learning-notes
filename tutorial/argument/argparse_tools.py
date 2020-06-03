import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def print_args(args):
    args = args.__dict__
    for k, v in args.items():
        print("{}: {}".format(k, v))


def get_parser():
    parser = argparse.ArgumentParser(description='for face verification train')
    parser.add_argument("-c", "--config", help="config file", type=str, default='config.py')
    parser.add_argument('--input_size', nargs='+',help="--input size 112,112", type=int,default=[112,112])
    # parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    # parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    # parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument('--flag', action='store_true', help='flag', default=False)
    parser.add_argument('--show', type=str2bool, help='Turn on/off show image', default=False)
    args = parser.parse_args()
    print_args(args)
    return args


if __name__ == '__main__':
    args = get_parser()
