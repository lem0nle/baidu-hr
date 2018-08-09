# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import json
import sys
import os
import six
from ast import literal_eval
import hr.document as doc
from hr.skill import doc_tagger
import hr.recommendation as rcm

postfile = json.load(open('data/campus_post.json'))
posts = {}
for p in postfile:
    posts[p['id']] = p

mcqs = doc.load('data/mc_questions_refined.json', doc.MCQuestion)
saqs = doc.load('data/sa_questions.json', doc.SAQuestion)

model_path = 'data/mc_clustering_model.pkl'
if os.path.exists(model_path):
    bank = rcm.QuestionBank(mcqs, saqs, model_path)
else:
    bank = rcm.QuestionBank(mcqs, saqs, keep_n=5)
    bank.save_model(model_path)


def rcm_api(postid, resume):
    post = doc.Post(id=postid, name=posts[postid]['name'],
                    terms=posts[postid]['terms'],
                    skills=posts[postid]['skills'])
    resume = doc.Resume(terms=resume, skills=doc_tagger.tag(resume))
    if postid == 6:
        if 'php后端' in resume.skills and 'c/c++' not in resume.skills:
            post.skills = [[s for s in inner if s != 'c/c++']
                           for inner in post.skills]
        if 'c/c++' in resume.skills and 'php后端' not in resume.skills:
            post.skills = [[s for s in inner if s != 'php后端']
                           for inner in post.skills]
    qs = bank.recommend(post, resume)
    return [q.id for q in qs]


if __name__ == '__main__':
    try:
        _, postid, resume = sys.argv
    except ValueError:
        print('Please input postid(int) and resume(string or list).')
        sys.exit(1)
    try:
        postid = int(postid)
    except ValueError:
        print('Postid should be int-type.')
        sys.exit(1)
    if six.PY2:
        resume = resume.decode(sys.stdin.encoding)
    if not resume.strip().startswith('['):
        resume = [resume]
    else:
        try:
            resume = literal_eval(resume)
        except (ValueError, SyntaxError):
            print('Resume should be string-type or list-type.')
            sys.exit(1)

    ques_ids = rcm_api(postid, resume)
    print(ques_ids)
