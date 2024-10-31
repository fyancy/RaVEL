import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def set_logger(filename='default.txt', add_flag=True, stream=sys.stdout):
    sys.stdout = Logger(filename, add_flag, stream)


def main():
    set_logger('b.txt')
    # sys.stderr = Logger("a.log", sys.stderr)     # redirect std err, if necessary
    # now it works
    print('print something')
    print("*" * 4)
    # sys.stdout.write("???")
    import time
    # time.sleep(10)
    print("other things")


if __name__ == '__main__':
    main()

