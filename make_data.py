import glob
import random
import os
import json
import ipdb
st = ipdb.set_trace
num_examples = 0
version = "v5"
pretrain = False


if pretrain:
    folder1= "/grogu/user/lilic/filtered_openwebmath/incontextv4_sft_train/*"
    folder2= "/home/mprabhud/datasets/o1/wikipedia_openwebmath/incontextv4_sft_train/*"
    store_folder = f"/home/mprabhud/datasets/o1/filtered_openwebmath/{version}"

    os.makedirs(store_folder,exist_ok=True)
    all_files = glob.glob(folder1) + glob.glob(folder2)
    all_data = []
    for file in all_files:
        data = json.load(open(file,'r'))
        all_data.extend(data)

    # Shuffle the data randomly
    random.shuffle(all_data)

    # Calculate split sizes (90% train, 10% test)
    train_size = int(0.95 * len(all_data))

    # Split the data
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # Save train data
    train_path = os.path.join(store_folder, f'train.json')
    if not os.path.exists(train_path):
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=4)
    else:
        print(f"File {train_path} already exists")

    # Save test data  
    test_path = os.path.join(store_folder, f'test.json')
    if not os.path.exists(test_path):
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=4)
    else:
        print(f"File {test_path} already exists")

    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(test_data)} test examples to {test_path}")
        
else:
    version = "v1"
    store_folder = f"/home/mprabhud/datasets/o1/math_hendrycks/{version}"
    os.makedirs(store_folder,exist_ok=True)
    folder = "/grogu/user/lilic/MATH/MATH/train/*"
    all_categories = glob.glob(folder)
    all_train_data = []
    for category in all_categories:
        print(category)
        all_files = glob.glob(f"{category}/*")
        for i, file in enumerate(all_files):
            print(f"{i}/{len(all_files)}")
            data = json.load(open(file,'r'))
            all_train_data.append(data)

    folder = "/grogu/user/lilic/MATH/MATH/test/*"
    all_test_data = []
    for category in all_categories:
        all_files = glob.glob(f"{category}/*")
        for i, file in enumerate(all_files):
            print(f"{i}/{len(all_files)}")
            data = json.load(open(file,'r'))
            all_test_data.append(data)


    # Save train data
    train_path = os.path.join(store_folder, f'train.json')
    with open(train_path, 'w') as f:
        json.dump(all_train_data, f, indent=4)

    # Save test data  
    test_path = os.path.join(store_folder, f'test.json')
    with open(test_path, 'w') as f:
        json.dump(all_test_data, f, indent=4)
