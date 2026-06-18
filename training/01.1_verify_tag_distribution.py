import json
from collections import Counter

courses = Counter()
modules = Counter()
langs = Counter()
combo = Counter()

with open('/srv/llama/training/extracted/raw_pages.jsonl') as f:
    for line in f:
        r = json.loads(line)
        courses[r['course']] += 1
        modules[f"{r['course']}/{r['module']}"] += 1
        langs[r['lang']] += 1
        combo[f"{r['course']}/{r['lang']}"] += 1

print('Pages per course:')
for k, v in courses.most_common():
    print(f'  {k}: {v}')
print()
print('Pages per language:')
for k, v in langs.most_common():
    print(f'  {k}: {v}')
print()
print('Course x Language:')
for k, v in sorted(combo.items()):
    print(f'  {k}: {v}')