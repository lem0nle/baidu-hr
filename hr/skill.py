# -*- coding: UTF-8 -*-
from __future__ import unicode_literals, print_function
import json
import random
import six.moves.cPickle as pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf
from .text import cut, Corpus
from .util import get_resource

label_list = ['c/c++', 'c#', 'python', 'go', 'java', '其他编程语言', '应用服务',
              'javascript', 'html/css', 'java后端', 'php后端', 'python后端',
              '数据库', '图形图像', '语音识别', 'nlp', '数据挖掘', '软件工程',
              '加密和安全', '算法设计', '数学基础', '计算机网络', '体系结构', '硬件',
              '操作系统', '分布式系统', '系统管理与维护', 'windows', 'ios',
              'android', '机器学习', '深度学习']


def _record_path(tree, path):
    # record the path of nodes in the skill_tree
    record = {}
    for child in tree:
        if type(child) == dict:
            for key in child:
                record[key.lower()] = path
                record.update(_record_path(child[key], [key.lower()] + path))
        else:
            record[child.lower()] = path
    return record


def _skill2vec(skills, k):
    return np.array([1 if label_list[k] in i else 0 for i in skills])


class TreeTagger:
    """
    """

    def __init__(self, tree):
        self.tree = tree
        self.record = _record_path(tree, [])

    def _find_label(self, word):
        word = word.lower()
        label = None
        if word in label_list:
            label = word
        elif word in self.record:
            for i in self.record[word]:
                if i.lower() in label_list:
                    label = i.lower()
                    break
        return label

    def tag(self, sentences, no_dup=True):
        skill_list = []
        for l in sentences:
            skills = list()
            text_split = cut(l)
            bigram = [text_split[i] + text_split[i + 1]
                      for i in range(len(text_split) - 1)]
            words = text_split + bigram
            for x in words:
                label = self._find_label(x)
                if label:
                    skills.append(label)
            if no_dup:
                skills = list(set(skills))
            skill_list.append(skills)
        return skill_list


class ClfTagger:
    """
    """

    def __init__(self, model=None, **kwargs):
        self._model = model
        self._model_args = kwargs
        self.models = []
        self.clf_thres = None
        self.corpus = None

    def tag(self, sentences):
        """
        """
        skill_list = [[]] * len(sentences)
        x = self.corpus.get_tfidf(sentences)
        for k in range(len(self.models)):
            y_pred = self.models[k].predict_proba(x)[:, 1:] > self.clf_thres[k]
            for i in range(len(sentences)):
                if y_pred[i]:
                    skill_list[i].append(label_list[k])
        return skill_list

    def train(self, sentences, skills, frac=0.6):
        data = list(zip(sentences, skills))
        random.shuffle(data)
        train_num = int(len(data) * frac)
        validation_data = data[:train_num // 5]
        train_data = data[train_num // 5:train_num]
        test_data = data[train_num:]

        train_sentences, _ = zip(*train_data)
        corpus = Corpus(train_sentences)
        self.corpus = corpus

        x_valid, y_valid = zip(*validation_data)
        x_valid = corpus.get_tfidf(x_valid)

        x_test, y_test = zip(*test_data)
        x_test = corpus.get_tfidf(x_test)

        thresholds = np.array(range(1, 11)) / 10

        for k in range(len(label_list)):
            self.models.append(self._model(**self._model_args))
            pos = [x for x in train_data if label_list[k] in x[1]]
            neg = [x for x in train_data if label_list[k] not in x[1]]
            if len(neg) > 2 * len(pos):
                neg = random.sample(neg, 2 * len(pos))
            k_data = pos + neg
            random.shuffle(k_data)
            x_k, y_k = zip(*k_data)
            x_k = corpus.get_tfidf(x_k)
            y_k = _skill2vec(y_k, k)
            self.models[k].fit(x_k, y_k)

            f1_max = 0
            clf_thres = []
            for t in thresholds:
                y_pred_k = self.models[k].predict_proba(x_valid)[:, 1:] > t
                y_valid_k = _skill2vec(y_valid, k)
                p, r, f1, _ = prf(y_valid_k, y_pred_k)
                if f1[1] > f1_max:
                    f1_max = f1[1]
                    thres_max = t
            clf_thres.append(thres_max)
        self.clf_thres = clf_thres

        p_list = []
        r_list = []
        f1_list = []
        for k in range(len(label_list)):
            y_pred = self.models[k].predict_proba(x_test)[:, 1:] > clf_thres[k]
            p, r, f1, _ = prf(_skill2vec(y_test, k), y_pred)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
        return p_list, r_list, f1_list

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump((self.corpus, self.models, self.clf_thres),
                        file, protocol=2)

    @classmethod
    def load(cls, filename):
        tagger = cls()
        tagger.corpus, tagger.models, tagger.clf_thres = \
            pickle.load(open(filename, "rb"))
        return tagger


_tr = json.loads(get_resource('resources/doc_skills.json'))
#: Tag documents according to skill tree.
doc_tagger = TreeTagger(_tr)

_tr = json.loads(get_resource('resources/doc_skills.json'))
#: Tag questions according to skill tree. The same as doc_tagger for now.
ques_tagger = TreeTagger(_tr)

if __name__ == '__main__':
    sentences = [
        "基于 node.js 的具有多人注册、登录、发表文章、登出等功能的简单博客 . 使用了 Express 框架和 ejs 模板引擎，以 Markdown 作为文章的编写格式 ， 数据库采用 Mongodb. MVC 模式开发.(个人项目)个人工作: 负责前台页面的设计与调整，后台文章管理、评论、关注等模块的功能实现.引入 pjax ，使用自定义代码块样式提高用户体验，为博客的实用性提供了保障.项目地址: http://chen.zhaishidan.cn/github地址: https://github.com/StionZhai/spring-blog (继续完善中 ， 已投入使用)",
        "该项目弥补了本学院网络答题的空白.前台使用 bootstrap3 框架构建，后台使用 java.共分为三个模块----学生模块、教师模块和管理员模块.学生模块主要功能为在线答题;教师模块提供试2题管理、试卷管理、判卷等功能;管理员模块主要负责对 1人员信息的管理.得到了老师们的好评.(三人小组，svn)个人工作: 负责前台页面的设计， ajax 与后台的数据交互.主要使用 js 对逻辑流程中的数据交互进行优化，明显提高了用户体验，减少了用户来回跳转页面和查询数据库的等待时间.项目地址:http://pan.baidu.com/s/1sjFbvdF (正在完善中，这里是一些前台草稿，quiz.rar)",
        "毕业设计管理系统 | 2014.02- 2014.03",
        "该项目意在方便教师对毕业生的毕业设计相关信息进行管理.分为开题、答辩、报表、系统和教师 5 个管理模块.使用 ajax 交互以提高用户体验，减少刷新给服务器带来的压力，使用状态位作为整个管理流程(各项功能的开闭)的控制器，采用 MVC 开发模式.前台基于 bootstrap3 框架.2015 年将投入使用(两人小组，svn)个人工作: 负责前台页面的设计与编写、一部分后台的功能模块，，以及 ajax 交互 . 独立解决了异步上传文件、给 n 多 DOM 元素注册互不干扰事件等问题，采用状态位动态修改 dom 元素，避免了局部刷新再次查询数据库 .实现了 SPA .项目地址 : http://pan.baidu.com/s/1sjFbvdF (GraduationProject.rar)",
        "仿 ThinkSNS微博系统 | 2013.07- 2013.08",
        "以 ThinkSNS 为原型设计的微博系统 ， 方便用户迅速找到感兴趣的人，方便用户间的互动和交流，还能给用户统一体验.后台使用 java + mysql ，整个项采用 MVC 开发模式.(10人，svn)个人工作:负责协调模块小组组员间配合，并参与了数据库设计以及后台接口的编写."
    ]

    print(doc_tagger.tag(sentences))
