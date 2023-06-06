"""Given descriptions gathered in files, extracts and processes chapters."""

import multiprocessing
import os
import pickle
import pandas as pd
from tqdm import tqdm
from chapter_utils import parse_timestamp, extract_timestamp, clean_str
from args import SSD_DIR, DATA_DIR

# Check already processed files
done_path = os.path.join(SSD_DIR, "chapters_chapters.csv")
if os.path.exists(done_path):
    done = set(pd.read_csv(done_path)['file_id'])
else:
    done = set()
descriptions_path = os.path.join(SSD_DIR, "chapters_descriptions")
files = os.listdir(descriptions_path)
print(len(files))
files = [x for x in files if x not in done]
print(len(files))

def desc2chapter(description):
    timestamp_lines = parse_timestamp(description)
    if len(timestamp_lines) > 1:  # more than a timestamp
        chapters = []
        for line in timestamp_lines:
            time_str, time, st, ed = extract_timestamp(line)
            if time == -1:
                continue
            title = line[:st] + line[ed:]
            title = clean_str(title)
            if title:
                chapters.append({'label': title, 'time': time})
        if len(chapters) > 1 and not all(len(y['label'].strip()) <= 1 for y in chapters)\
            and all(chapters[i]['time'] < chapters[i + 1]['time'] for i in range(len(chapters) - 1)):  # more than a timestamp post cleaning + avoid all 1-char titles + increasing timestamps
            return chapters
    return None

def process(z):
    """Given a (video_id, description), extracts and processes potential chapters from the description."""
    video_id, description = z[0], z[1]
    timeline = desc2chapter(str(description))
    if timeline is None:  # no chapter
        return None
    else:
        return {'video_id': video_id, 'chapters': timeline}

directory = os.path.join(DATA_DIR, "chapters_data")
inc = len(os.listdir(directory))
outfile = f'{directory}/{str(inc).zfill(4)}.pkl'
all_chapters = {}
for file in tqdm(files):
    cur_path = os.path.join(descriptions_path, file)
    df = pd.read_csv(cur_path)
    with multiprocessing.Pool(24) as p:
        results = p.map(process, [(x['video_id'], x['description']) for _, x in df.iterrows()])
    results = {x['video_id']: x['chapters'] for x in results if x is not None}
    all_chapters.update(results)
    done.add(file)
print(len(all_chapters))
print(sum(len(x) for x in all_chapters.values()))
pickle.dump(all_chapters, open(outfile, 'wb'))

done = pd.DataFrame({'file_id': list(done)}, columns=['file_id'])
print(len(done))
done.to_csv(done_path, index=False)