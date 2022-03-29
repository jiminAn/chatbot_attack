import sys
import random

input_file = sys.argv[1]

with open(input_file, 'r') as f:
	dataset = f.readlines()

random.shuffle(dataset) # suffle the dataset
cut_off = int(len(dataset)*0.1) 

# slice the list. 
# the last elements after the cut_off value will be the test set.
# the rest if for training

train_data, test_data = dataset[:-cut_off], dataset[-cut_off:]
train_path = input_file + '.train'
test_path = input_file + '.test'

with open(train_path, 'w') as f:
	f.write(''.join(train_data))

with open(test_path, 'w') as f:
	f.write(''.join(test_data))


'''
print(f'number of conversations in the data set: {len(dataset)}')
print(f'number of conversations in train set: {len(train_data)}')
print(f'number of conversations in eval set: {len(test_data)}')
'''
