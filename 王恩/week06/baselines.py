"""课程规则 / TF-IDF 路线，改为多语种分类。词表只在训练集拟合。"""
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


class ScriptRule:
    """文字系统 + 少量功能词；共享拉丁字母的语言是规则方法的难点。"""
    WORDS = {
        "en": "the a an is are was were i you he she it we they my your and of to with this that",
        "es": "el la los las una un es soy estoy que de y por para no con quiero tengo",
        "pt": "o a os as uma um é sou estou que de e por para não com você tenho",
        "tr": "bir bu ben sen için ve ile çok değil var ne benim mı mi onun",
        "vi": "tôi bạn là của và không có một người những chúng được đã đang",
    }
    PATTERNS = [("hi", r"[\u0900-\u097f]"), ("th", r"[\u0e00-\u0e7f]"),
                ("ar", r"[\u0600-\u06ff]"), ("ru", r"[\u0400-\u04ff]"),
                ("ko", r"[\uac00-\ud7af\u1100-\u11ff]"),
                ("ja", r"[\u3040-\u30ff]"), ("zh", r"[\u3400-\u9fff]")]

    def fit(self, texts, labels):
        self.majority = Counter(labels).most_common(1)[0][0]
        self.labels = set(labels)
        return self

    def predict(self, texts):
        predictions = []
        for text in texts:
            counts = [(label, len(re.findall(pattern, text))) for label, pattern in self.PATTERNS
                      if label in self.labels]
            # 含假名时优先判日文，即使汉字更多。
            if "ja" in self.labels and re.search(r"[\u3040-\u30ff]", text):
                predictions.append("ja")
                continue
            best = max(counts, key=lambda item: item[1], default=(self.majority, 0))
            if best[1]:
                predictions.append(best[0])
                continue
            words = re.findall(r"[^\W\d_]+", text.casefold())
            scores = {label: sum(word in vocab.split() for word in words)
                      for label, vocab in self.WORDS.items() if label in self.labels}
            maximum = max(scores.values(), default=0)
            winners = [label for label, score in scores.items() if score == maximum]
            predictions.append(winners[0] if maximum and len(winners) == 1 else self.majority)
        return predictions


def build_baseline(method, seed):
    if method == "rule":
        return ScriptRule()
    classifiers = {
        "tfidf_nb": MultinomialNB(alpha=0.1),
        "tfidf_lr": LogisticRegression(C=4.0, max_iter=1000, random_state=seed),
        "tfidf_svm": LinearSVC(C=1.0, random_state=seed),
    }
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                        max_features=100000, sublinear_tf=True), classifiers[method])
