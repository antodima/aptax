from pathlib import Path

import grain.python as grain


def create_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0,
):
    estimated_batches = len(dataset) // batch_size
    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=worker_count,
    )

    return dataloader, estimated_batches


def load_stories(eot_token="<|endoftext|>", max_stories=1000):
    """Efficiently load stories from a text file.
    Each story ends with <|endoftext|>.
    """
    file_path = Path("aptax/data/tinystories/TinyStories-1000.txt")
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading stories from {file_path}...")
    stories = []
    current_story = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                for part in parts[:-1]:
                    current_story.append(part)
                    story_text = "".join(current_story).strip()
                    if story_text:
                        stories.append(story_text + eot_token)
                        if max_stories and len(stories) >= max_stories:
                            break
                    current_story = []
                if parts[-1].strip():
                    current_story = [parts[-1]]
                else:
                    current_story = []
                if max_stories and len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)
        if current_story and (not max_stories or len(stories) < max_stories):
            story_text = "".join(current_story).strip()
            if story_text:
                stories.append(story_text + eot_token)

    print(f"Loaded {len(stories):,} stories")
    return stories


class TextsDataset:
    def __init__(self, texts, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.eot_token = "<|endoftext|>"
        self.special_tokens = {self.eot_token}
        self.eot_token_id = tokenizer.encode(
            self.eot_token, allowed_special=self.special_tokens
        )[0]
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        story = self.texts[idx]
        tokens = self.tokenizer.encode(story, allowed_special=self.special_tokens)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        tokens.extend([self.eot_token_id] * (self.max_seq_len - len(tokens)))
        return tokens
