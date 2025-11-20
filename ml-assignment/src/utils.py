def read_file(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return f.read()
