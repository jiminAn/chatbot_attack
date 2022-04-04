import module
from collections import defaultdict
import torch
assert torch.cuda.is_available()

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
model = module.DialogElectra()

test_file = './data/wellness.user_chatbot.16k.test'
category_idx_file = './data/model9_category_idx'
tests = list()

categories = defaultdict(str)
with open(category_idx_file, 'rt', encoding='UTF8') as f:
	for line in f.readlines():
		category, idx = line.split('\t')
		categories[int(idx)] = category.strip()

print('user','user prediction','chatbot','chatbot prediction', sep='\t')
with open(test_file, 'rt', encoding='UTF8') as f:
	for line in f.readlines():
		user, chatbot = line.split('\t')
		user, chatbot = user.strip(), chatbot.strip()
		user_predict, chatbot_predict = model.predict(user), model.predict(chatbot)
		if user_predict == 3 or chatbot_predict == 3:
			print(user, chatbot, categories[user_predict], categories[chatbot_predict], sep='\t')