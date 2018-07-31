import json
import hr.document as doc
import hr.recommendation as rcm

file = open('data/examples/post_campus.json')
post_resume = json.load(file)

mcqs = doc.load('data/mc_questions.json', doc.MCQuestion)
saqs = doc.load('data/sa_questions.json', doc.SAQuestion)
questions = rcm.QuestionBank(mcqs, saqs)

for v in post_resume.values():
    post = doc.Post(terms=v['post'])
    resumes = v['resume']
    resumes = [doc.Resume(terms=r) for r in resumes]
    for resume in resumes:
        print(questions.recommend(post, resume))
