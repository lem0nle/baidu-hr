from random import sample, choice, shuffle
from functools import reduce
from collections import defaultdict, Counter
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

    post_skills = reduce(lambda a, b: a + b, post_skills)
    resume_skills = reduce(lambda a, b: a + b, resume_skills)

    rand_skill = choice(resume_skills)
    post_cnt = Counter(post_skills)
    reg_post = {k: v / sum(post_cnt.values()) for k, v in post_cnt.items()}
    res_cnt = Counter(resume_skills)
    reg_res = {k: v / sum(res_cnt.values()) for k, v in res_cnt.items()}

    weights = merge_weights(reg_post, reg_res)

    return weights, rand_skill


class QuestionBank:
    def __init__(self, mc_ques, sa_ques):
        # group questions by skill
        mcq_dict = defaultdict(list)
        saq_dict = defaultdict(list)
        for ques in mc_ques:
            for skill in ques.skills:
                mcq_dict[skill].append(ques)
        for ques in sa_ques:
            for skill in ques.skills:
                saq_dict[skill].append(ques)

        self.mcq_dict = mcq_dict
        self.saq_dict = saq_dict

    def recommend(self, post, resume, num=8):
        weights, rand_skill = _get_weight(post, resume)

        counts = {k: round(v * num) for k, v in weights.items()}
        counts = list(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        ques_list = []
        if rand_skill in self.saq_dict:
            ques_list.append(choice(self.saq_dict[rand_skill]))
        for k, v in counts:
            if v == 0:
                break
            ques_list.extend(sample(self.mcq_dict[k], v))
        if len(ques_list) < num:
            tmp = []
            n_remain = num - len(ques_list)
            for k, v in counts[:3]:
                tmp.extend(sample(self.mcq_dict[k],
                                  min(n_remain, len(self.mcq_dict[k]))))
            ques_list.extend(sample(tmp, n_remain))

        return ques_list
