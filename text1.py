# 一.文本处理的基本方法

# import jieba             
# content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
# 1.1 精确模式分词，试图将句子最精确地切开，适合文本分析
# 将返回一个生成器对象
# print(jieba.cut(content,cut_all=False))
# 若需直接返回列表内容, 使用jieba.lcut即可
# print(jieba.lcut(content,cut_all=False))

# 1.2 全模式分词，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能消除 歧义.
# 将返回一个生成器对象
# print(jieba.cut(content,cut_all=True))
# 若需直接返回列表内容, 使用jieba.lcut即可
# print(jieba.lcut(content,cut_all=True))

# 1.3 搜索引擎模式分词，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
# 将返回一个生成器对象
# print(jieba.cut_for_search(content))
# 若需直接返回列表内容, 使用jieba.lcut_for_search即可,区别是对'女干事', '交换机'等较长词汇都进行了再次分词.
# print(jieba.lcut_for_search(content))

# 1.4 中文繁体分词:针对中国香港, 台湾地区的繁体文本进行分词.
# content = "煩惱即是菩提，我暫且不提"
# print(jieba.lcut(content))

# 1.5 使用用户自定义词典,userdict.txt是自定义的文件
# content="八一双鹿更名为八一南昌篮球队"
# 使用了用户自定义词典后的结果:
# jieba.load_userdict("./userdict.txt")
# print(jieba.lcut(content))

# 1.6 词性标注就是标注出一段文本中每个词汇的词性.
# import jieba.posseg as pseg
# print(pseg.lcut("我爱北京天安门")) 

# -------------------------------------------
# 二.文本张量表示方法

# 2.1 one-hot词向量表示
# 2.1.1 onehot编码实现
# 导入用于对象保存与加载的joblib
import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    # 初始化一个全为0的向量
    zero_list = [0]*len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1   
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)


# 2.1.2 onehot编码器的使用:
t = joblib.load(tokenizer_path)
# 编码token为"李宗盛"
token = "李宗盛"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0]*len(vocab)
# 令zero_List的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list) 
