import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--words', dest='words')
parser.add_argument('--hello', dest='hello')


args = parser.parse_args()

print(args)

print(args.hello)
