import os
import os.path as osp
import random
import math

input_path = "/mnt/data/hcj/atlas/corpora/wiki/enwiki-dec2018"
output_path = "/mnt/data/hcj/atlas/corpora/wiki/enwiki-dec2018_sampled"
file_name_1 = "infobox.jsonl"
file_name_2 = "text-list-100-sec.jsonl"

rate = 0.1

infobox_sampled = None
with open(osp.join(input_path,file_name_1), encoding="utf-8") as f:
    lines_1 = f.readlines()
    infobox_sampled = random.sample(lines_1, math.ceil(rate*len(lines_1)))
    
with open (osp.join(output_path,"infobox_{}sampled.jsonl".format(rate)), "w") as f:
    f.writelines(infobox_sampled)
    
text_sampled = None
with open(osp.join(input_path,file_name_2), encoding="utf-8") as f:
    lines_2 = f.readlines()
    text_sampled = random.sample(lines_2, math.ceil(rate*len(lines_2)))
    
with open (osp.join(output_path,"text-list-100-sec_{}sampled.jsonl".format(rate)), "w") as f:
    f.writelines(text_sampled)
    
# os.system('DATA_DIR="/mnt/data/hcj/atlas"')
# os.system('TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018_sampled/text-list-100-sec_{}sampled.jsonl"'.format(rate*100))
# os.system('INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018_sampled/infobox_{}sampled.jsonl"'.format(rate*100))

# os.system('shuf ${TEXTS} > "${TEXTS}.shuf"')
# os.system('head -n 2000 "${TEXTS}.shuf" | head -n 1000 > "${TEXTS}.shuf.test"')
# os.system('head -n 2000 "${TEXTS}.shuf" | tail -n 1000 > "${TEXTS}.shuf.valid"')
# os.system('tail -n +2000 "${TEXTS}.shuf" > "${TEXTS}.shuf.train"')

# os.system('shuf ${INFOBOXES} > "${INFOBOXES}.shuf"')
# os.system('head -n 2000 "${INFOBOXES}.shuf" | head -n 1000 > "${INFOBOXES}.shuf.test"')
# os.system('head -n 2000 "${INFOBOXES}.shuf" | tail -n 1000 > "${INFOBOXES}.shuf.valid"')
# os.system('tail -n +2000 "${INFOBOXES}.shuf" > "${INFOBOXES}.shuf.train"')