class Colors():
    def __init__(self):
        self.HEADER = '\033[95m'
        self.STATUS = '\033[94m'
        self.OK = '\033[92m'
        self.WARNING = '\033[93m'
        self.ERROR = '\033[91m'
        self.ENDCOLOR = '\033[0m'

    def print_ok(self, text):
        print(self.STATUS + text + self.ENDCOLOR)

    def print_info(self, text):
        print(self.HEADER + text + self.ENDCOLOR)

    def print_status(self, text):
        print(self.OK + text + self.ENDCOLOR)

    def print_warning(self, text):
        print(self.WARNING + text + self.ENDCOLOR)

    def print_error(self, text):
        print(self.ERROR + text + self.ENDCOLOR)

    def print_test(self):
        print(self.HEADER + 'Header' + self.ENDCOLOR)
        print(self.STATUS + 'Status' + self.ENDCOLOR)
        print(self.OK + 'OK' + self.ENDCOLOR)
        print(self.WARNING + 'Warning' + self.ENDCOLOR)
        print(self.ERROR + 'Error' + self.ENDCOLOR)
