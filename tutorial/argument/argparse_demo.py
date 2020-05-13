import argparse


# python train.py -net mobilefacenet -b 200 -w 4

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='for face verification train')
    parser.add_argument("-c", "--config", help="config file", default='config.py', type=str)
    # parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    # parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    # parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    # parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    # parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    # parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    # parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    parser.add_argument('--show', default=True, action='store_false', help='Bool type')
    parser.add_argument('--flag', type=str2bool, nargs='?', const=True, help='Turn on or turn off flag')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    with open(args.config) as f:
        source = f.read()
        executable = compile(source, 'source.py', 'exec')
        dd = eval(executable)
        print(dd)
