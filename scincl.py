import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

# Load the abstracts from the pickle file
with open('article_abstracts.pkl', 'rb') as f:
    abstracts = pickle.load(f)

# Load the CSV file for fine-tuning SciNCL
lcr_data = pd.read_csv('lcr_input_final.csv')

# Fine-tune the SciNCL model
def fine_tune_scincl(contexts, abstracts):

    labels = [1] * len(contexts)  # All pairs are considered matching for this task

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("malteos/scincl")
    model = AutoModelForSequenceClassification.from_pretrained("malteos/scincl")

    # Create the dataset
    data = {'context': contexts, 'abstract': abstracts, 'label': labels}
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["context"], examples["abstract"], truncation=True, max_length=512, padding="max_length",)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    train_dataset, eval_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine-tuned-scincl")
    tokenizer.save_pretrained("./fine-tuned-scincl")

    return SentenceTransformer("./fine-tuned-scincl")

# Prepare the data for fine-tuning
contexts = lcr_data['context'].tolist()
quoted_abstracts = lcr_data['abstract'].tolist()

contexts = [str(context) for context in contexts]
quoted_abstracts = [str(abstract) for abstract in quoted_abstracts]

# Fine-tune the SciNCL model
scincl_model = fine_tune_scincl(contexts, quoted_abstracts)
