import sys


class Logger(object):
    def __init__(self, file_path, also_print_to_terminal):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf8')
        self.print_to_terminal = also_print_to_terminal

    def write(self, message):
        if self.print_to_terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def redirect_print_to_file(file_path, print_to_terminal=False):
    sys.stdout = Logger(file_path, print_to_terminal)


def print_argparse(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
