class Table:
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            f.close()
            # assert(lines[0] == "# epics ioc card slot link farow mml1\n")
            names = lines[0].split()[1:]
            cols = {}
            for name in names:
                cols[name] = []
            for line in lines[1:]:
                row = line.split()
                for name, var in zip(names, row, strict=True):
                    cols[name].append(var)
            self.names = names
            self.cols = cols

    def find(self, name, key):
        return self.cols[name].index(key)
