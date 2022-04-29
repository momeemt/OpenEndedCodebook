import argparse


def get_mcc_args():
    parser = argparse.ArgumentParser(
        description='boostrap maze MCC experiment'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help='experiment name'
    )
    parser.add_argument(
        '-b', '--boostrap',
        default='default', type=str,
        help='boostrap name to use'
    )
    parser.add_argument(
        '-g', '--generation',
        default=2000, type=int,
        help='generation num'
    )
    parser.add_argument(
        '--agent-criteria',
        default=1, type=int,
        help='minimum criteria of agent'
    )
    parser.add_argument(
        '--maze-criteria',
        default=1, type=int,
        help='minimum criteria of maze'
    )
    parser.add_argument(
        '--agent-pop',
        default=250, type=int,
        help='max agent population size'
    )
    parser.add_argument(
        '--maze-pop',
        default=50, type=int,
        help='max maze population size'
    )
    parser.add_argument(
        '--agent-limit',
        default=0, type=int,
        help='resource limit of agent'
    )
    parser.add_argument(
        '--maze-limit',
        default=5, type=int,
        help='recourse limit of maze'
    )
    parser.add_argument(
        '--agent-batch',
        default=40, type=int,
        help='batch size of agent'
    )
    parser.add_argument(
        '--maze-batch',
        default=10, type=int,
        help='batch size of maze'
    )
    parser.add_argument(
        '--num-cores',
        default=4, type=int,
        help='num of multiprocesses'
    )

    args = parser.parse_args()

    if args.name==None:
        args.name = args.boostrap

    return args


def get_boostrap_args():
    parser = argparse.ArgumentParser(
        description='boostrap maze MCC experiment'
    )

    parser.add_argument(
        '-n', '--name',
        default='default', type=str,
        help='experiment name'
    )
    parser.add_argument(
        '--agent-num',
        default=20, type=int,
        help='number to prepare agent'
    )
    parser.add_argument(
        '--maze-num',
        default=10, type=int,
        help='number to prepare maze'
    )
    parser.add_argument(
        '--wall-gene-num',
        default=2, type=int,
        help='number of wall gene'
    )
    parser.add_argument(
        '--path-gene-num',
        default=2, type=int,
        help='number of path gene'
    )
    parser.add_argument(
        '--num-cores',
        default=4, type=int,
        help='num of multiprocesses'
    )

    args = parser.parse_args()

    return args
