import jieba
import jieba.posseg as pseg
from io import StringIO
from gensim import corpora
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel
from .util import get_resource_filename
from nltk.tokenize import WordPunctTokenizer
tk = WordPunctTokenizer()

jieba.load_userdict(get_resource_filename('resources/dict'))


def _cut_zh(text, filter_pos=False, filter_marks=False):
    """Cut text.

    Args:
        text(str): text to cut

    Returns:
        list: word list
    """
    marks = {' ', '-', '/', '，', '：', ';', '、', '；', '（', '）', ',', "。"}
    if filter_pos:
        words = [x.word.replace(" ", "") for x in pseg.cut(text)
                 if x.flag in ["eng", "n", "nz", "nr"]]
    else:
        words = jieba.lcut(text)
    if filter_marks:
        words = [x for x in words if x not in marks]
    return words


def cut(text):
    rv = []
    buffer = StringIO()
    mode = None
    for c in text.lower():
        if ord(c) < 128:
            if mode == 'zh':
                rv.extend(_cut_zh(buffer.getvalue()))
                buffer = StringIO()
            mode = 'en'
            buffer.write(c)
        else:
            if mode == 'en':
                rv.extend(tk.tokenize(buffer.getvalue()))
                buffer = StringIO()
            mode = 'zh'
            buffer.write(c)
    if mode == 'zh':
        rv.extend(_cut_zh(buffer.getvalue()))
    elif mode == 'en':
        rv.extend(tk.tokenize(buffer.getvalue()))
    return rv


def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000 or num == 0xa0:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


def clean(text):
    text = full2half(text)
    text = text.strip().strip('、').replace('\r\n', '\n')
    return text


class Corpus:
    """Class Corpus."""

    def __init__(self, sentences, keep_n=100000):
        """
        Args:
            sentences(list): list of str
            keep_n(int): maximum length of dictionary (default: 100000)
        """
        self.sentences = [cut(x) for x in sentences]
        self.dictionary = corpora.Dictionary(self.sentences)
        self.dictionary.filter_extremes(no_below=0, keep_n=keep_n)
        self.tfidf_model = TfidfModel(self.dictionary.doc2bow(x)
                                      for x in self.sentences)

    def get_bow(self, sentences=None, dense=True):
        """Get the Bag-Of-Word representation of sentences.

        Args:
            sentences(list): list of str (optional)
            dense(bool): whether to convert corpus to dense matrix (default:
                         ``True``)

        Returns:
            list or numpy.ndarray:
                numpy matrix if dense, else sparse representation
        """
        if sentences is None:
            sentences = self.sentences
        else:
            sentences = [cut(x) for x in sentences]
        vec = [self.dictionary.doc2bow(x) for x in sentences]
        if dense:
            vec = corpus2dense(vec, len(self.dictionary)).transpose()
        return vec

    def get_tfidf(self, sentences=None, dense=True):
        """Get the TF-IDF representation of sentences.

        Args:
            sentences(list): list of str (optional)
            dense(bool): whether to convert corpus to dense matrix (default:
                         ``True``)

        Returns:
            list or numpy.ndarray:
                numpy matrix if dense, else sparse representation
        """
        if sentences is None:
            sentences = self.sentences
        else:
            sentences = [cut(x) for x in sentences]
        vec = [self.dictionary.doc2bow(x) for x in sentences]
        vec = self.tfidf_model[vec]
        if dense:
            vec = corpus2dense(vec, len(self.dictionary)).transpose()
        return vec

    def __getstate__(self):
        return {
            'sentences': [],
            'dictionary': self.dictionary,
            'tfidf_model': self.tfidf_model
        }


if __name__ == '__main__':
    sentences = [
        "基于 node.js 的具有多人注册、登录、发表文章、登出等功能的简单博客 . 使用了 Express 框架和 ejs 模板引擎，以 Markdown 作为文章的编写格式 ， 数据库采用 Mongodb. MVC 模式开发.(个人项目)个人工作: 负责前台页面的设计与调整，后台文章管理、评论、关注等模块的功能实现.引入 pjax ，使用自定义代码块样式提高用户体验，为博客的实用性提供了保障.项目地址: http://chen.zhaishidan.cn/github地址: https://github.com/StionZhai/spring-blog (继续完善中 ， 已投入使用)",
        "该项目弥补了本学院网络答题的空白.前台使用 bootstrap3 框架构建，后台使用 java.共分为三个模块----学生模块、教师模块和管理员模块.学生模块主要功能为在线答题;教师模块提供试2题管理、试卷管理、判卷等功能;管理员模块主要负责对 1人员信息的管理.得到了老师们的好评.(三人小组，svn)个人工作: 负责前台页面的设计， ajax 与后台的数据交互.主要使用 js 对逻辑流程中的数据交互进行优化，明显提高了用户体验，减少了用户来回跳转页面和查询数据库的等待时间.项目地址:http://pan.baidu.com/s/1sjFbvdF (正在完善中，这里是一些前台草稿，quiz.rar)",
        "毕业设计管理系统 | 2014.02- 2014.03",
        "该项目意在方便教师对毕业生的毕业设计相关信息进行管理.分为开题、答辩、报表、系统和教师 5 个管理模块.使用 ajax 交互以提高用户体验，减少刷新给服务器带来的压力，使用状态位作为整个管理流程(各项功能的开闭)的控制器，采用 MVC 开发模式.前台基于 bootstrap3 框架.2015 年将投入使用(两人小组，svn)个人工作: 负责前台页面的设计与编写、一部分后台的功能模块，，以及 ajax 交互 . 独立解决了异步上传文件、给 n 多 DOM 元素注册互不干扰事件等问题，采用状态位动态修改 dom 元素，避免了局部刷新再次查询数据库 .实现了 SPA .项目地址 : http://pan.baidu.com/s/1sjFbvdF (GraduationProject.rar)",
        "仿 ThinkSNS微博系统 | 2013.07- 2013.08",
        "以 ThinkSNS 为原型设计的微博系统 ， 方便用户迅速找到感兴趣的人，方便用户间的互动和交流，还能给用户统一体验.后台使用 java + mysql ，整个项采用 MVC 开发模式.(10人，svn)个人工作:负责协调模块小组组员间配合，并参与了数据库设计以及后台接口的编写."
    ]
    # corpus = Corpus(sentences)
    # print(corpus.get_tfidf(sentences[:2]))
    print(cut(sentences[0]))
