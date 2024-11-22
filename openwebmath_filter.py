from datasets import load_dataset, load_from_disk, Dataset

# ds = load_from_disk("openwebmath_train")

ds = load_dataset("open-web-math/open-web-math")

split_ds = ds["train"].train_test_split(test_size=0.2) # for some reason everything is in ds["train"]
train_ds = split_ds["train"]
test_ds = split_ds["test"]

import json

# Filter and save train dataset in chunks
CHUNK_SIZE = 100000
current_chunk = []
chunk_num = 0
print(len(train_ds))

for i, item in enumerate(train_ds):
    if i % 10000 == 0:
        print(i, len(current_chunk))
    metadata = json.loads(item['metadata'])
    math_score = metadata['extraction_info']['math_score']
    # if math_score > 0.8:
    if 'wikipedia' in item['url']:
        current_chunk.append(item)
        
    if len(current_chunk) >= CHUNK_SIZE or (i == len(train_ds) - 1 and current_chunk):  # Save smaller chunks to be safe
        chunk_dataset = Dataset.from_list(current_chunk)
        # chunk_dataset.save_to_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_{chunk_num}")
        chunk_dataset.save_to_disk(f"/grogu/user/lilic/wikipedia_openwebmath/train/chunk_{chunk_num}")
        print(f"Saved train chunk {chunk_num} with {len(current_chunk)} items")
        current_chunk = []
        chunk_num += 1

# Filter and save test dataset in chunks
current_chunk = []
chunk_num = 0

for i, item in enumerate(test_ds):
    metadata = json.loads(item['metadata'])
    math_score = metadata['extraction_info']['math_score']
    # if math_score > 0.8:
    if 'wikipedia' in item['url']:
        current_chunk.append(item)
        
    if len(current_chunk) >= CHUNK_SIZE or (i == len(test_ds) - 1 and current_chunk):
        chunk_dataset = Dataset.from_list(current_chunk)
        # chunk_dataset.save_to_disk(f"/grogu/user/lilic/filtered_openwebmath/test/chunk_{chunk_num}")
        chunk_dataset.save_to_disk(f"/grogu/user/lilic/wikipedia_openwebmath/test/chunk_{chunk_num}")
        print(f"Saved test chunk {chunk_num} with {len(current_chunk)} items")
        current_chunk = []
        chunk_num += 1


print(f"Original train size: {len(train_ds)}")
print(f"Original test size: {len(test_ds)}")