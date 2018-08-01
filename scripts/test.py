# -*- coding: UTF-8 -*-
from __future__ import unicode_literals, print_function
import json
import os
from six import itervalues
import hr.document as doc
import hr.recommendation as rcm

file = open('data/examples/post_campus.json')
post_resume = json.load(file)

mcqs = doc.load('data/mc_questions.json', doc.MCQuestion)
saqs = doc.load('data/sa_questions.json', doc.SAQuestion)

model_path = 'data/mc_clustering_model.pkl'
if os.path.exists(model_path):
    bank = rcm.QuestionBank(mcqs, saqs, model_path)
else:
    bank = rcm.QuestionBank(mcqs, saqs)
    bank.save_model(model_path)

for v in itervalues(post_resume):
    post = doc.Post(terms=v['post'])
    resumes = v['resume']
    resumes = [doc.Resume(terms=r) for r in resumes]
    for resume in resumes:
        print(bank.recommend(post, resume))
