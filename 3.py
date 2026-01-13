import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations


df = pd.read_csv("ap_dataset.csv")
df["data_bon"] = pd.to_datetime(df["data_bon"])

df = df[df["retail_product_name"].notna()]
df = df[df["retail_product_name"] != "Packaging"]

price_mean = df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()

baskets = (
    df.groupby("id_bon")["retail_product_name"]
      .apply(list)
      .tolist()
)

baskets = [b for b in baskets if len(b) >= 2]


rng = np.random.default_rng(42)
idx = np.arange(len(baskets))
rng.shuffle(idx)

split = int(0.8 * len(idx))
train_baskets = [baskets[i] for i in idx[:split]]
test_baskets  = [baskets[i] for i in idx[split:]]


class CoocNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.item_count = defaultdict(int)
        self.pair_count = defaultdict(int)
        self.vocab = []
        self.v2i = {}

    def fit(self, baskets):
        for b in baskets:
            s = set(b)
            for p in s:
                self.item_count[p] += 1
            for a, c in combinations(sorted(s), 2):
                self.pair_count[(a, c)] += 1
                self.pair_count[(c, a)] += 1

        self.vocab = sorted(self.item_count.keys())
        self.v2i = {p:i for i,p in enumerate(self.vocab)}
        self.N = len(baskets)
        self.V = len(self.vocab)
        return self

    def log_prior(self, p):
        return np.log((self.item_count[p] + self.alpha) / (self.N + self.alpha * self.V))

    def log_p_i_given_p(self, i, p):
        co = self.pair_count.get((i, p), 0)
        return np.log((co + self.alpha) / (self.item_count[p] + self.alpha * self.V))

    def rank(self, basket_partial, candidates=None, forbid_in_basket=True, use_revenue=True, topk=10):
        B = set(basket_partial)

        if candidates is None:
            candidates = self.vocab

        scores = []
        for p in candidates:
            if forbid_in_basket and p in B:
                continue

            lp = self.log_prior(p)
            for i in B:
                if i in self.item_count:
                    lp += self.log_p_i_given_p(i, p)

            if use_revenue:
                pr = price_mean.get(p, 0.0)
                if pr > 0:
                    lp += np.log(pr)
                else:
                    lp += -50
            scores.append((lp, p))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scores[:topk]]

model = CoocNB(alpha=1.0).fit(train_baskets)


def make_leave_one_out_examples(baskets, seed=123, max_examples=None):
    rng = np.random.default_rng(seed)
    ex = []
    for b in baskets:
        s = list(dict.fromkeys(b))
        if len(s) < 2:
            continue
        y = rng.choice(s)
        partial = [x for x in s if x != y]
        ex.append((partial, y))
    if max_examples:
        ex = ex[:max_examples]
    return ex

test_ex = make_leave_one_out_examples(test_baskets, seed=7)

def hit_at_k(examples, k):
    hits = 0
    for partial, y in examples:
        recs = model.rank(partial, topk=k, use_revenue=True)
        hits += int(y in recs)
    return hits / len(examples)

for k in [1, 3, 5]:
    print(f"Hit@{k}: {hit_at_k(test_ex, k):.4f}")


pop_rank = sorted(model.item_count.items(), key=lambda x: x[1], reverse=True)
pop_list = [p for p,_ in pop_rank]

revenue_sum = df.groupby("retail_product_name")["SalePriceWithVAT"].sum().to_dict()
rev_list = sorted(revenue_sum.keys(), key=lambda p: revenue_sum[p], reverse=True)

cos_partial = [
    "Crazy Schnitzel",
    "Crazy Sauce"
]
print("Pentru cosul: ", cos_partial, " avem recomandarile: ")
print(model.rank(cos_partial))

def baseline_hit_at_k(examples, ranked_list, k, forbid_in_basket=True):
    hits = 0
    for partial, y in examples:
        B = set(partial)
        recs = []
        for p in ranked_list:
            if forbid_in_basket and p in B:
                continue
            recs.append(p)
            if len(recs) == k:
                break
        hits += int(y in recs)
    return hits / len(examples)

for k in [1,3,5]:
    print(f"[Baseline Popularity]  Hit@{k}: {baseline_hit_at_k(test_ex, pop_list, k):.4f}")
    print(f"[Baseline Revenue]     Hit@{k}: {baseline_hit_at_k(test_ex, rev_list, k):.4f}")
