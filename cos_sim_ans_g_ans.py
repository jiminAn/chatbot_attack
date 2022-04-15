import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util
import time
assert torch.cuda.is_available()
start = time.time()  # 시작 시간 저장
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
embedder = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

test_set = []
test_set_path = "./dialogLM/data/wellness.user_chatbot.16k.test"
with open(test_set_path, 'rt', encoding="UTF8") as f:
    for line in f.readlines():
        q, a = line.split('\t')
        test_set.append((q.strip(), a.strip()))

gen_ans_data_path = "./sentenceBERT/data/q.generate_ans"
g_answers = []
with open(gen_ans_data_path, 'rt', encoding="UTF8") as f:
    for line in f.readlines():
        g_answers.append(line.strip())

start = time.time()
cos_sims = []
for test, g_ans in zip(test_set,g_answers):
    q, ans = test[0], test[1]
    len_ans = len(ans)
    g_ans_r = g_ans[:len_ans]
    ans_emb = embedder.encode(ans)
    g_ans_emb = embedder.encode(g_ans_r)
    cos_simirality = cos_sim(ans_emb, g_ans_emb)
    print(q, ans, g_ans_r, cos_simirality, sep='\t')
    cos_sims.append(cos_simirality)

print(time.time()-start)
df = pd.DataFrame(cos_sims)
sns.displot(df[0],kde=False)
plt.show()
