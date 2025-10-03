import pandas as pd
from collections import defaultdict, deque
import os

in_path = "results_and_data/early_exit_sft_dataset/test/theory_of_mind.csv"
out_path = "results_and_data/early_exit_sft_dataset/test/theory_of_mind_roundrobin.csv"

# Load
df = pd.read_csv(in_path)

# Basic validation
required_cols = {"story", "question"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"CSV must contain columns {required_cols}, but is missing {missing}. Columns present: {list(df.columns)}")

df = df.reset_index().rename(columns={"index": "_orig_idx"})

first_idx_per_story = df.groupby("story")["_orig_idx"].min().sort_values()
ordered_story_keys = list(first_idx_per_story.index)

story_to_rows = {}
for story in ordered_story_keys:
    rows = df[df["story"] == story].sort_values("_orig_idx")
    story_to_rows[story] = deque(rows.index.tolist())  # store row indices into df

ordered_indices = []
active_stories = ordered_story_keys.copy()

while active_stories:
    next_active = []
    for story in active_stories:
        qrows = story_to_rows[story]
        if qrows:
            ordered_indices.append(qrows.popleft())
            if qrows:  # still has more questions
                next_active.append(story)
    active_stories = next_active

out_df = df.loc[ordered_indices].drop(columns=["_orig_idx"]).reset_index(drop=True)

out_df.to_csv(out_path, index=False)

out_path
