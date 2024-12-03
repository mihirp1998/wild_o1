import glob
import os
import ipdb
st = ipdb.set_trace
all_md_files = glob.glob("/grogu/user/mprabhud/papers_o1/attention_papers/*/main.md")
print(len(all_md_files))
all_headings = []
for idx,md_file in enumerate(all_md_files):
    if os.path.exists(md_file):
        print(f'{idx}/{len(all_md_files)}')
        with open(md_file, "r") as f:
            content = f.read()
        # print(content)
        # Extract headings from markdown content
        headings = []
        for line in content.split('\n')[1:]:
            # Match lines starting with # symbols
            if line.strip().startswith('#'):
                headings.append(line.strip())
        print(f"\nHeadings in {md_file}:")
        all_headings.append(headings)
print('done')
