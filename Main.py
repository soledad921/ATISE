
# Some libraries to import
import argparse

from Train import train

parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    default = 'LinkPrediction',
    type = str,
    help='choose different tasks')

parser.add_argument(
    '--model',
    default='ATISE',type=str,
    help='choose models')

parser.add_argument(
    '--dataset',
    default='icews14',
    type=str, help='dataset to train on')

parser.add_argument(
    '--max_epoch',
    default=5000, type=int,
    help='number of total epochs (min value: 500)')

parser.add_argument(
    '--dim',
    default=500, type=int,
    help='number of dim')

parser.add_argument(
    '--batch',
    default=512, type=int,
    help='number of batch size')

parser.add_argument(
    '--lr',
    default=0.1, type=float,
    help='number of learning rate')

parser.add_argument(
    '--gamma',
    default=1, type=float,
    help='number of margin')

parser.add_argument(
    '--eta',
    default=10, type=int,
    help='number of negative samples per positive')

parser.add_argument(
    '--timedisc',
    default=0, type=int,
    help='method of discretizing time intervals')

parser.add_argument(
    '--cuda',
    default=True, type=bool,
    help='use cuda or cpu')

parser.add_argument(
    '--loss',
    default='logloss', type=str,
    help='loss function')

parser.add_argument(
    '--cmin',
    default=0.005, type=float,
    help='cmin')

parser.add_argument(
    '--gran',
    default=1, type=int,
    help='time unit for ICEWS datasets')

parser.add_argument(
    '--thre',
    default=1, type=int,
    help='the mini threshold of time classes in yago and wikidata')

def main(args):
    print(args)
    train(task=args.task,
          modelname=args.model,
          data_dir=args.dataset,
          dim=args.dim,
          batch=args.batch,
          lr =args.lr,
          max_epoch=args.max_epoch,
          gamma = args.gamma,
          lossname = args.loss,
          negsample_num=args.eta,
          timedisc = args.timedisc,
          cuda_able = args.cuda,
          cmin = args.cmin,
          gran = args.gran,
          count = args.thre
          )              


if __name__ == '__main__':
    main(parser.parse_args())
