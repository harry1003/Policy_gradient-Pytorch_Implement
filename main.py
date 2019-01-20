import argparse

from agent import Agent


def parse():
    parser = argparse.ArgumentParser(description="Policy gradient for pong-v0")
    parser.add_argument('--train', action="store_true", help='train mode')
    parser.add_argument('--test', action="store_true", help='test mode')
    parser.add_argument('-e', '--epochs', default=10000, help='training epochs')
    parser.add_argument('-b', '--batch_size', default=10, help='update model every batch_size of game')
    parser.add_argument('-l', '--lr', default=1e-4, help='learning rate')
    parser.add_argument('-g', '--gamma', default=0.99, help='discount reward ratio')
    parser.add_argument('-opt', '--optimizer', default="Adam", help='optimizer')
    parser.add_argument('--comet', action="store_true", help='visuilize package')

    """ fix checkpoint"""
    parser.add_argument('-c', '--checkpoint', default="", help='update model every batch_size of game')
    args = parser.parse_args()
    return args


def main(args):
    agent = Agent(args)
    
    if args.train:
        print("training mode")
        agent.train(args)

    if args.test:
        print("testing mode")
        '''
        fix load
        '''
        agent.load()
        agent.test()


if __name__ == '__main__':
    args = parse()
    main(args)
