import argparse

from utilities.Timer import Timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for utility module of split model inference')
    parser.add_argument('-path', metavar='base-path', action='store', default="./", required=False,
                        help='The base path for the project')
    parser.add_argument('-timer1', metavar='timer1-name', action='store', default="timer1", required=False,
                        help='The filename of timer 1')
    parser.add_argument('-timer2', metavar='timer2-name', action='store', default="timer2", required=False,
                        help='The filename of timer 2')
    args = parser.parse_args()

    timer1: Timer = Timer('timer 1', args.timer1).load()
    timer2: Timer = Timer('timer 2', args.timer2).load()

    difference = timer1.find_difference(timer2)
    running_latency_sum = 0
    counter = 0
    for key in difference.keys():
        counter += 1
        running_latency_sum += difference[key]
    running_latency_sum /= (1.0 * counter)

    print('Total latency is {}'.format(running_latency_sum))
