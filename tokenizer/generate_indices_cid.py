import argparse


# this file for generating CID proposed in MBGen
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='Bili_Cartoon')


    # hyper-param
    parser.add_argument('--k_base', type=int, default=64)

