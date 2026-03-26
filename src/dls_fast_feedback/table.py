class table:
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            f.close()
            # assert(lines[0] == "# epics ioc card slot link farow mml1\n")
            names = lines[0].split()[1:]
            cols = {}
            for n in names:
                cols[n] = []
            for l in lines[1:]:
                row = l.split()
                for n, v in zip(names, row):
                    cols[n].append(v)
            self.names = names
            self.cols = cols

    def find(self, name, key):
        return self.cols[name].index(key)
