import pickle
from random import sample, choice, shuffle
from itertools import chain
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from .text import Corpus
from .skill import doc_tagger


def merge_weights(dict1, dict2, frac=0.67):
    if not dict2:
        return dict1
    else:
        dict1 = {k: dict1[k] * frac for k in dict1}
        dict2 = {k: dict2[k] * (1 - frac) for k in dict2}
        return dict(Counter(dict1) + Counter(dict2))


def _get_weight(post, resume):
    post_skills = doc_tagger.tag(post.terms)
    resume_skills = doc_tagger.tag(resume.terms)

    # TODO: reduce weights after 优先

    post_skills = list(chain(*post_skills))
    resume_skills = list(chain(*resume_skills))
    if len(resume_skills) == 0:
        print(resume)

    rand_skill = choice(resume_skills)
    post_cnt = Counter(post_skills)
    reg_post = {k: v / sum(post_cnt.values()) for k, v in post_cnt.items()}
    res_cnt = Counter(resume_skills)
    reg_res = {k: v / sum(res_cnt.values()) for k, v in res_cnt.items()}

    weights = merge_weights(reg_post, reg_res)

    return weights, rand_skill


class QuestionBank:
    def __init__(self, mc_ques, sa_ques, model=None):
        sentences = [q.main + ';' + ','.join(q.options) for q in mc_ques]

        if model is None:
            # question clustering
            # TODO: set proper keep_n for your need
            corpus = Corpus(sentences, keep_n=100)
            # TODO: check model and n_clusters param
            model = KMeans(len(mc_ques) // 4, n_jobs=-1)
            model.fit(corpus.get_tfidf(sentences))
            labels = model.labels_
        else:
            if isinstance(model, str):
                corpus, model = pickle.load(open(model, 'rb'))
            labels = model.predict(corpus.get_tfidf(sentences))

        self.corpus = corpus
        self.model = model

        ques_by_cluster = defaultdict(list)
        for i, q in zip(labels, mc_ques):
            ques_by_cluster[i].append(q)
        self.mc_ques_clusters = list(ques_by_cluster.values())

        # build sa_dict in advance as it doesn't change
        self.sa_ques = sa_ques
        self.saq_dict = defaultdict(list)
        for ques in sa_ques:
            for skill in ques.skills:
                self.saq_dict[skill].append(ques)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.corpus, self.model), f)

    def recommend(self, post, resume, num=8):
        mc_ques = [choice(x) for x in self.mc_ques_clusters]

        # group mc questions by skill
        mcq_dict = defaultdict(list)
        for ques in mc_ques:
            for skill in ques.skills:
                mcq_dict[skill].append(ques)

        weights, rand_skill = _get_weight(post, resume)

        counts = {k: round(v * num) for k, v in weights.items()}
        counts = list(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        ques_list = []
        if rand_skill in self.saq_dict:
            ques_list.append(choice(self.saq_dict[rand_skill]))
        for k, v in counts:
            if v == 0:
                break
            ques_list.extend(sample(mcq_dict[k], min(v, len(mcq_dict[k]))))
        if len(ques_list) < num:
            tmp = []
            n_remain = num - len(ques_list)
            for k, v in counts[:3]:
                tmp.extend(sample(mcq_dict[k],
                                  min(n_remain, len(mcq_dict[k]))))
            ques_list.extend(sample(tmp, n_remain))

        return ques_list
