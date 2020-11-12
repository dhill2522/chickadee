class Resource(object):
    def __init__(self, name):
        super(Resource, self).__init__()
        self.name = name

    def __repr__(self):
        return self.name