import glob
import os
import json
import ipdb
st = ipdb.set_trace
num_examples = 0
version = "v5"
check_folder = "/grogu/user/lilic/filtered_openwebmath/incontextv4_sft_train/*"

for file in glob.glob(check_folder):
    data = json.load(open(file,'r'))
    num_examples += len(data)
print(num_examples)
