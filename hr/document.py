# -*- coding: UTF-8 -*-
from __future__ import unicode_literals, print_function
import json
import re
from uuid import uuid4
from .text import clean
from .skill import ques_tagger

_r_numword = re.compile(r'^\d[.、 ]([^0-9])')


class Document:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.__class__.__name__ + ": " \
            + str(self.__dict__)

    def __repr__(self):
        return str(self)


class MCQuestion(Document):
    @classmethod
    def build(cls, main, options, answer, skills, relabel=True):
        main = clean(main)
        main = _r_numword.sub(r'\1', main).strip()
        if isinstance(answer, list):
            new_answer = []
            for a in answer:
                new_answer.extend(_find_answer(a, options))
            answer = new_answer
        elif isinstance(answer, str):
            answer = _find_answer(answer, options)
        answer.sort()

        options = [clean(o) for o in options]
        options = [_r_numword.sub(r'\1', o).strip() for o in options]

        q = cls(id=str(uuid4()), main=main, options=options,
                answer=answer, skills=skills)
        if relabel:
            q._relabel()
        return q

    def _relabel(self):
        text = self.main + ';' + ','.join(self.options)
        skills = ques_tagger.tag([text])[0]
        self.skills.extend(skills)


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


def _find_answer(answer, options):
    if answer in options:
        return [options.index(answer)]
    try:
        answer = float(answer)
        options = [float(o) for o in options]
        return [options.index(answer)]
    except ValueError:
        pass
    answer = answer.lower()
    if all('a' <= x <= 'f' for x in answer):
        return [ord(x) - ord('a') for x in answer]
    return []


if __name__ == '__main__':
    # posts = load('data/posts.json', Post)
    # print(posts[0])
    # print(posts[0].terms)
    q = MCQuestion.build('1. 下列哪个是答案', ['1', '3', '4', '2'], '2', [])
    print(q)
    q = MCQuestion.build('1. 下', ['1', '3', '4', '2'], ['2', '4'], [])
    print(q)
    q = MCQuestion.build('1. 下', ['1', '3', '4', '2'], ['A', 'C'], [])
    print(q)
    q = MCQuestion.build('机器学习', ['1', '3', '4', '2'], 'ABC', [])
    print(q)
    q = MCQuestion.build('\n\n 1、 下', ['1', '3', '4', '2'], '2.0', [])
    print(q)
