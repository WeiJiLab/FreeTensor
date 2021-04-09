import itertools

def with_line_no(s: str):
    lines = list(s.splitlines())
    maxNuLen = len(str(len(lines)))
    fmt = "{:%dd}" % maxNuLen
    return "\n".join(map(
        lambda arg: '\033[33m' + fmt.format(arg[1] + 1) + '\033[0m' + ' ' + arg[0],
        zip(lines, itertools.count())))

