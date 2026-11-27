"""基于训练集词典和字符级 CRF 的抽取基线。"""

import re
import unicodedata
from collections import Counter, defaultdict

from data_sources import span


def event_prediction(event_type, trigger, args):
    return [{"type": event_type, "trigger": trigger, "args": args}]


class LexiconExtractor:
    """优先匹配较长的触发词，在类型限定的词典中查找论元。"""

    def __init__(self, rows, schema):
        triggers = defaultdict(Counter)
        arguments = defaultdict(lambda: defaultdict(Counter))
        for row in rows:
            event = row["event"]
            event_type = event["type"]
            triggers[event["trigger"]["text"]][event_type] += 1
            for argument in event["args"]:
                arguments[event_type][argument["text"]][argument["role"]] += 1

        self.trigger_labels = {
            value: counts.most_common(1)[0] for value, counts in triggers.items()
        }
        self.trigger_pattern = self.pattern(self.trigger_labels)
        self.argument_labels = {
            event_type: {
                value: counts.most_common(1)[0][0]
                for value, counts in arguments[event_type].items() if sum(counts.values()) >= 2
            }
            for event_type in schema
        }
        self.argument_patterns = {
            event_type: self.pattern(labels) for event_type, labels in self.argument_labels.items()
        }

    @staticmethod
    def pattern(labels):
        values = sorted(labels, key=lambda value: (-len(value), value))
        return re.compile("|".join(map(re.escape, values))) if values else None

    def predict(self, text):
        def rank(match):
            value = match.group()
            return len(value), self.trigger_labels[value][1], -match.start()

        trigger = max(self.trigger_pattern.finditer(text), key=rank, default=None)
        if trigger is None:
            return []
        event_type = self.trigger_labels[trigger.group()][0]
        pattern = self.argument_patterns[event_type]
        args = [
            dict(span(match.group(), match.start()), role=self.argument_labels[event_type][match.group()])
            for match in pattern.finditer(text)
        ] if pattern else []
        return event_prediction(event_type, span(trigger.group(), trigger.start()), args)


def to_bio(length, spans):
    tags = ["O"] * length
    # 单层 BIO 中优先保留长论元。触发词由另一套标签建模。
    for role, start, end in sorted(spans, key=lambda item: (item[1] - item[2], item[1], item[0])):
        if all(tag == "O" for tag in tags[start:end]):
            tags[start:end] = [f"B:{role}"] + [f"I:{role}"] * (end - start - 1)
    return tags


def from_bio(text, tags):
    parts = []
    i = 0
    while i < len(tags):
        if tags[i] == "O":
            i += 1
            continue
        role = tags[i][2:]
        end = i + 1
        while end < len(tags) and tags[end] == f"I:{role}":
            end += 1
        parts.append(dict(span(text[i:end], i), role=role))
        i = end
    return parts


def char_features(text, event_type=None, trigger=None):
    chars = ["<BOS>"] * 2 + list(text) + ["<EOS>"] * 2
    shape = [unicodedata.category(char) if len(char) == 1 else char for char in chars]
    for i in range(2, len(chars) - 2):
        window = chars[i - 2:i + 3]
        features = {f"char{j - 2}": value for j, value in enumerate(window)}
        features.update({
            "prev2": window[0] + window[1],
            "left": window[1] + window[2],
            "right": window[2] + window[3],
            "next2": window[3] + window[4],
            "shape": "/".join(shape[i - 1:i + 2]),
            "position": str(min(4, (i - 2) * 5 // len(text))),
        })
        if event_type is not None:
            features["type"] = event_type
            features["typed_char"] = event_type + "|" + window[2]
        if trigger:
            distance = i - 2 - trigger["start"]
            if distance >= 0:
                distance = max(0, i - 1 - trigger["end"])
            features["trigger_offset"] = str(min(10, max(-10, distance)))
            features["trigger_region"] = "before" if distance < 0 else "after" if distance > 0 else "inside"
        yield features


class CRFExtractor:
    """句级类型分类、字符级触发词标注、按类型训练的论元标注。"""

    def __init__(self, rows, schema, seed):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC
        from sklearn_crfsuite import CRF

        def train_crf(features, tags):
            return CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=60).fit(features, tags)

        types = [row["event"]["type"] for row in rows]
        self.only_type = types[0] if len(set(types)) == 1 else None
        if self.only_type is None:
            self.type_model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True),
                LinearSVC(C=2.0, random_state=seed),
            ).fit([row["text"] for row in rows], types)

        trigger_features, trigger_tags = [], []
        role_features, role_tags = defaultdict(list), defaultdict(list)
        for row in rows:
            text, event = row["text"], row["event"]
            event_type, trigger = event["type"], event["trigger"]
            trigger_features.append(list(char_features(text, event_type)))
            trigger_tags.append(to_bio(len(text), [("trigger", trigger["start"], trigger["end"])]))
            if schema[event_type]:
                role_features[event_type].append(list(char_features(text, trigger=trigger)))
                role_tags[event_type].append(to_bio(len(text), [
                    (a["role"], a["start"], a["end"]) for a in event["args"]
                ]))
        self.trigger_model = train_crf(trigger_features, trigger_tags)
        self.role_models = {
            event_type: train_crf(features, role_tags[event_type])
            for event_type, features in role_features.items()
        }

    def predict(self, text):
        event_type = self.only_type or str(self.type_model.predict([text])[0])
        tags = self.trigger_model.predict_single(list(char_features(text, event_type)))
        candidates = from_bio(text, tags)
        trigger = max(candidates, key=lambda part: len(part["text"]), default=None)
        if trigger is not None:
            trigger.pop("role")
        role_model = self.role_models.get(event_type)
        args = []
        if role_model:
            tags = role_model.predict_single(list(char_features(text, trigger=trigger)))
            args = from_bio(text, tags)
        return event_prediction(event_type, trigger or {}, args)
