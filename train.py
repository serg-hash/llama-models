import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_spanish_dataset(name: str, split: str = "train", text_column: str = "text"):
    """Load a dataset from the HF Hub and keep Spanish examples."""
    ds = load_dataset(name, split=split)
    lang_cols = [c for c in ["language", "lang"] if c in ds.column_names]
    if lang_cols:
        col = lang_cols[0]
        ds = ds.filter(lambda x: str(x[col]).startswith("es"))
    if text_column not in ds.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    return ds.map(lambda x: {"text": x[text_column]})


def tokenize_dataset(tokenizer, ds):
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True)

    return ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    ds = load_spanish_dataset(args.dataset_name, text_column=args.text_column)
    tokenized = tokenize_dataset(tokenizer, ds)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=500,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama model on a Spanish dataset")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B", help="Base model identifier")
    parser.add_argument("--dataset-name", default="OpenAssistant/oasst1", help="Dataset name on the HF Hub")
    parser.add_argument("--text-column", default="text", help="Column containing the text")
    parser.add_argument("--output-dir", default="finetuned_model", help="Where to save the fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    main(parser.parse_args())
