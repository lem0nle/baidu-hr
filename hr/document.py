import json


class Document:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.__class__.__name__ + ": " \
            + str(self.__dict__)

    def __repr__(self):
        return str(self)


class MCQuestion(Document):

    # build的时候记得先clean，全角转半角之类的

    pass


class SAQuestion(Document):
    pass


class Post(Document):
    pass


class Resume(Document):
    pass


def dump(docs, filename):
    docs = [x.__dict__ for x in docs]
    with open(filename, 'w') as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)


def load(filename, cls):
    docs = json.load(open(filename))
    return [cls(**x) for x in docs]


if __name__ == '__main__':
    posts = load('data/posts.json', Post)
    print(posts[0])
    print(posts[0].terms)
