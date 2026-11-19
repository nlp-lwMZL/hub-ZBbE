"""L1 · TF-IDF + 逻辑回归 / SVM。运行: python l1_tfidf.py

分词方式、分类器和类别权重在文件顶部设置。
"""
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from common import DATA_DIR, report

ANALYZER = "word"  # "word"=按词，"char"=按字
MODEL = "lr"  # "lr"=逻辑回归，其他值=SVM
CLASS_WEIGHT = None  # 可改为 "balanced" 使用类别权重

if ANALYZER == "word":
    NGRAM = (1, 2)
else:
    NGRAM = (2, 3)  # 按字时使用二元和三元特征


def tokenize(text):
    """word 模式切成空格分隔的词串；char 模式原样返回（交给 vectorizer 按字）"""
    if ANALYZER == "word":
        # lcut 返回分词列表，join 用空格连接，供 sklearn 按词提取特征。
        return " ".join(jieba.lcut(text))
    return text


def read_tsv(path):
    """读 TSV，返回 (文本列表, 标签列表)，跳过表头"""
    rows = list(csv.reader(open(path, encoding="utf-8"), delimiter="\t"))[1:]
    texts = []
    labels = []
    for label, text in rows:
        labels.append(label)
        texts.append(text)
    return texts, labels


def main():
    train_texts, y_train = read_tsv(DATA_DIR / "train.tsv")
    test_texts, y_true = read_tsv(DATA_DIR / "test.tsv")

    X_train = []
    for text in train_texts:
        X_train.append(tokenize(text))
    X_test = []
    for text in test_texts:
        X_test.append(tokenize(text))

    # ngram_range=(1, 2) 使用单词及相邻词对；min_df=2 过滤只出现一次的特征。
    # sublinear_tf 用对数缩放词频；fit_transform 学习词表并生成稀疏矩阵。
    vectorizer = TfidfVectorizer(analyzer=ANALYZER, ngram_range=NGRAM, min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    # 测试集沿用训练集词表，避免泄漏测试信息。
    X_test_vec = vectorizer.transform(X_test)
    print(f"特征维度: {X_train_vec.shape[1]}")

    if MODEL == "lr":
        # max_iter 是优化迭代次数上限；balanced 权重会提高少数类的重要程度。
        clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT)
    else:
        clf = LinearSVC(class_weight=CLASS_WEIGHT)
    # fit(特征, 标签) 训练；predict(特征) 返回按输入顺序排列的预测标签。
    clf.fit(X_train_vec, y_train)

    print(f"\nanalyzer={ANALYZER}  model={MODEL}  class_weight={CLASS_WEIGHT}\n")
    y_pred = clf.predict(X_test_vec)
    return report(y_true, y_pred)


if __name__ == "__main__":
    from common import prepare_quick_data
    prepare_quick_data()
    main()
