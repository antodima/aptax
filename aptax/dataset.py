import json
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp

from aptax.sampling import sample_from_bnn, sample_from_scm


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


def load_squad(max_records=None):
    dataset = []
    file_path = Path("aptax/data/squad/train-v2.0.json")
    with open(file_path, "r") as f:
        squad_data = json.load(f)
        for data in squad_data["data"]:
            for paragraph in data["paragraphs"]:
                qas = paragraph["qas"][0]
                question = qas["question"]
                if len(qas["answers"]) == 0:
                    answer = qas["plausible_answers"][0]["text"]
                else:
                    answer = qas["answers"][0]["text"]
                dataset.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )

    if max_records is not None:
        return dataset[:max_records]
    return dataset


def get_loss_mask(
    token_ids: list,
    answer_start_idx: int,
    padding_token_id: int = 50256,
):
    seq_len = len(token_ids)
    indices = jnp.arange(seq_len)
    is_answer = indices >= (answer_start_idx - 1)
    is_not_padding = jnp.array(token_ids) != padding_token_id
    loss_mask = jnp.where(is_answer & is_not_padding, 1, 0)
    return loss_mask


class TextsDataset:
    def __init__(self, texts, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eos_token
        self.eot_token_id = tokenizer.encode(self.eot_token)[0]
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        story = self.texts[idx]
        story_ids = self.tokenizer.encode(story)
        if len(story_ids) > self.max_seq_len:
            story_ids = story_ids[: self.max_seq_len]

        inputs_ids = story_ids
        inputs_ids.extend([self.eot_token_id] * (self.max_seq_len - len(inputs_ids)))
        labels_ids = jnp.array(inputs_ids)[1:].tolist() + [self.eot_token_id]
        loss_mask = get_loss_mask(
            labels_ids,
            answer_start_idx=0,
            padding_token_id=self.eot_token_id,
        ).tolist()

        return {
            "text": story,
            "inputs": inputs_ids,
            "labels": labels_ids,
            "loss_mask": loss_mask,
        }


class QADataset:
    def __init__(self, data, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.eot_token = tokenizer.eos_token
        self.eot_token_id = tokenizer.encode(self.eot_token)[0]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        prompt = f"Question: {question}\nAnswer: "
        answer = item["answer"]

        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
        input_ids.extend([self.eot_token_id] * (self.max_seq_len - len(input_ids)))
        labels_ids = jnp.array(input_ids)[1:].tolist() + [self.eot_token_id]

        # loss_mask = get_loss_mask(
        #     labels_ids,
        #     answer_start_idx=len(prompt_ids),
        #     padding_token_id=self.eot_token_id,
        # ).tolist()
        loss_mask = get_loss_mask(
            labels_ids,
            answer_start_idx=0,
            padding_token_id=self.eot_token_id,
        ).tolist()

        return {
            "text": prompt + answer,
            "inputs": input_ids,
            "labels": labels_ids,
            "loss_mask": loss_mask,
        }


def generate_synthetic_tabular_data(key, n_samples, n_features, n_classes):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # 1. Decide which prior to use (50/50 mix)
    X, y = None, None
    if jax.random.uniform(k1) > 0.5:
        # SCM Prior Logic
        # - Sample a DAG structure
        # - Define non-linear functions for each node
        # - Sample noise and propagate through the graph
        X, y = sample_from_scm(k2, n_samples, n_features)
    else:
        # BNN Prior Logic
        # - Sample random weights and architecture
        # - Generate x, compute y = BNN(x)
        X, y = sample_from_bnn(k3, n_samples, n_features)

    # 2. Convert continuous y to multi-class labels
    # Sample N_c - 1 unique values from y as boundaries
    shuffled_y = jax.random.permutation(k4, y)
    boundaries = jnp.sort(shuffled_y[: n_classes - 1])

    # Map continuous values to discrete bins
    y_discrete = jnp.digitize(y, boundaries)
    # 3. Shuffle class labels to remove ordering
    unique_labels = jnp.unique(y_discrete)
    shuffled_labels = jax.random.permutation(k5, unique_labels)
    indices = jnp.searchsorted(unique_labels, y_discrete)
    y_final = shuffled_labels[indices]

    return X, y_final


class TablesDataset:
    def __init__(self, n_tables, n_samples, n_features, n_classes, key=None):
        if key is None:
            key = jax.random.PRNGKey(42)
        self.tables = []
        for i in range(n_tables):
            k_table = jax.random.fold_in(key, i)
            X, y = generate_synthetic_tabular_data(
                k_table, n_samples, n_features, n_classes
            )
            self.tables.append({"X": X, "y": y})

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        table = self.tables[idx]
        return {"inputs": table["X"], "labels": table["y"]}
