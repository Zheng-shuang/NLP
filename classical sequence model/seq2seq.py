# 使用seq2seq模型架构实现英译法任务

# 第一步: 导入必备的工具包
# 从io工具包导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 用于正则表达式
import re
# 用于随机生成数据
import random
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化方法工具包
from torch import optim
# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 第二步: 对持久化文件中数据进行处理, 以满足模型训练要求
# 2.1 将指定语言中的词汇映射成数值:
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

class Lang:
    def __init__(self, name):
        """初始化函数中参数name代表传入某种语言的名字"""
        # 将name传入类中
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典, 其中0，1对应的SOS和EOS已经在里面了
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的自然数索引，这里从2开始，因为0，1已经被开始和结束标志占用了
        self.n_words = 2  

    def addSentence(self, sentence):
        """添加句子函数, 即将句子转化为对应的数值序列, 输入参数sentence是一条句子"""
        # 根据一般国家的语言特性(我们这里研究的语言都是以空格分个单词)
        # 对句子进行分割，得到对应的词汇列表
        for word in sentence.split(' '):
            # 然后调用addWord进行处理
            self.addWord(word)

    def addWord(self, word):
        """添加词汇函数, 即将词汇转化为对应的数值, 输入参数word是一个单词"""
        # 首先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 如果不在, 则将这个词加入其中, 并为它对应一个数值，即self.n_words
            self.word2index[word] = self.n_words
            # 同时也将它的反转形式加入到self.index2word中
            self.index2word[self.n_words] = word
            # self.n_words一旦被占用之后，逐次加1, 变成新的self.n_words
            self.n_words += 1
# # 实例化参数:
# name = "eng"
# # 输入参数:
# sentence = "hello I am Jay"
# # 调用:
# engl = Lang(name)
# engl.addSentence(sentence)
# print("word2index:", engl.word2index)
# print("index2word:", engl.index2word)
# print("n_words:", engl.n_words)

# 2.2 字符规范化:
# 将unicode转为Ascii, 我们可以认为是去掉一些语言中的重音标记：Ślusàrski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """字符串规范化函数, 参数s代表传入的字符串"""
    # 使字符变为小写并去除两侧空白符, z再使用unicodeToAscii去掉重音标记
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# # 输入参数:
# s = "Are you kidding me?"
# # 调用:
# nsr = normalizeString(s)
# print(nsr)

# 2.3 将持久化文件中的数据加载到内存, 并实例化类Lang
data_path = 'data/eng-fra.txt'
def readLangs(lang1, lang2):
    """读取语言函数, 参数lang1是源语言的名字, 参数lang2是目标语言的名字
       返回对应的class Lang对象, 以及语言对列表"""
    # 从文件中读取语言对并以/n划分存到列表lines中
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，并以\t进行再次划分, 形成子列表, 也就是语言对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines] 
    # 然后分别将语言名字传入Lang类中, 获得对应的语言对象, 返回结果
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

# 输入参数:
lang1 = "eng"
lang2 = "fra"
# 调用:
input_lang, output_lang, pairs = readLangs(lang1, lang2)
# print("input_lang:", input_lang)
# print("output_lang:", output_lang)
# print("pairs中的前五个:", pairs[:5])

# 2.4 过滤出符合我们要求的语言对:
# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10

# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    """语言对过滤函数, 参数p代表输入的语言对, 如['she is afraid.', 'elle malade.']"""
    # p[0]代表英语句子，对它进行划分，它的长度应小于最大长度MAX_LENGTH并且要以指定的前缀开头
    # p[1]代表法文句子, 对它进行划分，它的长度应小于最大长度MAX_LENGTH
    return len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes) and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    """对多个语言对列表进行过滤, 参数pairs代表语言对组成的列表, 简称语言对列表"""
    # 函数中直接遍历列表中的每个语言对并调用filterPair即可
    return [pair for pair in pairs if filterPair(pair)]

# 调用:
# fpairs = filterPairs(pairs)
# print("过滤后的pairs前五个:", fpairs[:5])

# 2.5 对以上数据准备函数进行整合, 并使用类Lang对语言对进行数值映射:
def prepareData(lang1, lang2):
    """数据准备函数, 完成将所有字符串数据向数值型数据的映射以及过滤语言对
       参数lang1, lang2分别代表源语言和目标语言的名字"""
    # 首先通过readLangs函数获得input_lang, output_lang对象，以及字符串类型的语言对列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 对字符串类型的语言对列表进行过滤操作
    pairs = filterPairs(pairs)
    # 对过滤后的语言对列表进行遍历
    for pair in pairs:
        # 并使用input_lang和output_lang的addSentence方法对其进行数值映射
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # 返回数值映射后的对象, 和过滤后语言对
    return input_lang, output_lang, pairs

# 调用:
input_lang, output_lang, pairs = prepareData('eng', 'fra')
# print("input_n_words:", input_lang.n_words)
# print("output_n_words:", output_lang.n_words)
# print(random.choice(pairs))

# 2.6 将语言对转化为模型输入需要的张量:
def tensorFromSentence(lang, sentence):
    """将文本句子转换为张量, 参数lang代表传入的Lang的实例化对象, sentence是预转换的句子"""
    # 对句子进行分割并遍历每一个词汇, 然后使用lang的word2index方法找到它对应的索引
    # 这样就得到了该句子对应的数值列表
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 然后加入句子结束标志
    indexes.append(EOS_token)
    # 将其使用torch.tensor封装成张量, 并改变它的形状为nx1, 以方便后续计算
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    """将语言对转换为张量对, 参数pair为一个语言对"""
    # 调用tensorFromSentence分别将源语言和目标语言分别处理，获得对应的张量表示
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    # 最后返回它们组成的元组
    return (input_tensor, target_tensor)

# 输入参数:
# 取pairs的第一条
pair = pairs[0]
# 调用:
pair_tensor = tensorsFromPair(pair)
# print(pair_tensor)



# 第三步: 构建基于GRU的编码器和解码器

# 3.1 构建基于GRU的编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """它的初始化参数有两个, input_size代表解码器的输入尺寸即源语言的
            词表大小，hidden_size代表GRU的隐层节点数, 也代表词嵌入维度, 同时又是GRU的输入尺寸"""
        super(EncoderRNN, self).__init__()
        # 将参数hidden_size传入类中
        self.hidden_size = hidden_size
        # 实例化nn中预定义的Embedding层, 它的参数分别是input_size, hidden_size
        # 这里的词嵌入维度即hidden_size
        # nn.Embedding的演示在该代码下方,得到embedded
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 然后实例化nn中预定义的GRU层, 它的参数是hidden_size
        # nn.GRU的演示在该代码下方
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """编码器前向逻辑函数中参数有两个, input代表源语言的Embedding层输入张量
           hidden代表编码器层gru的初始隐层张量"""
        # 将输入张量进行embedding操作, 并使其形状变为(1,1,-1),-1代表自动计算维度
        # 理论上，我们的编码器每次只以一个词作为输入, 因此词汇映射后的尺寸应该是[1, embedding]
        # 而这里转换成三维的原因是因为torch中预定义gru必须使用三维张量作为输入, 因此我们拓展了一个维度
        output = self.embedding(input).view(1, 1, -1)
        # 然后将embedding层的输出和传入的初始hidden作为gru的输入传入其中, 
        # 获得最终gru的输出output和对应的隐层张量hidden， 并返回结果
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        # 将隐层张量初始化成为1x1xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 实例化参数:
hidden_size = 25
input_size = 20
# 输入参数:
# pair_tensor[0]代表源语言即英文的句子，pair_tensor[0][0]代表句子中的第一个单词。这里指的是单词的张量
input = pair_tensor[0][0]
# 初始化第一个隐层张量，1x1xhidden_size的0张量,三维张量
hidden = torch.zeros(1, 1, hidden_size)
# 调用:
encoder = EncoderRNN(input_size, hidden_size)
encoder_output, hidden = encoder(input, hidden)
print(encoder_output)
