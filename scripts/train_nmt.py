#!/usr/bin/env python3
import argparse, torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType

def main(args):
    DATA = Path("data"); MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"✓ models/ dir created: {MODEL_DIR.resolve()}")
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # TODO: implement dataset loading
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.add_token:
        tokenizer.add_tokens([f">>{args.lang}<<"])
        model.resize_token_embeddings(len(tokenizer))
    lora = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.SEQ_2_SEQ_LM,
                      target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR/f"rendille-{args.lang}"),
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, fp16=torch.cuda.is_available(),
        save_steps=args.save_steps, eval_steps=200, logging_steps=50, predict_with_generate=True,
    )
    # Data collator for seq2seq (new in Transformers 4.35+)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Load datasets from prepared text files
    from datasets import load_dataset
    
    def load_split(split_file):
        """Load a parallel text split with 'src ||| tgt' format."""
        dataset = load_dataset(
            "text",
            data_files={"text": str(DATA / split_file)},
            split="train"
        )
        # Parse parallel lines: "Rendille text ||| English text"
        def parse_pair(example):
            src, tgt = example["text"].split("|||")
            return {"src": src.strip(), "tgt": tgt.strip()}
        
        dataset = dataset.map(parse_pair, desc=f"Parsing {split_file}")
        # Tokenize for NLLB
        def preprocess(example):
            # NLLB expects input_ids and labels
            # For fine-tuning on parallel text: input = source, labels = target
            src_tok = tokenizer(
                example["src"],
                truncation=True,
                max_length=200,
                padding="max_length"
            )
            tgt_tok = tokenizer(
                example["tgt"],
                truncation=True,
                max_length=200,
                padding="max_length"
            )
            return {
                "input_ids": src_tok["input_ids"],
                "attention_mask": src_tok["attention_mask"],
                "labels": tgt_tok["input_ids"],
            }
        
        tokenized = dataset.map(
            preprocess,
            remove_columns=["text", "src", "tgt"],
            desc=f"Tokenizing {split_file}"
        )
        return tokenized

    print("Loading datasets...")
    train_dataset = load_split("train.txt")
    eval_dataset = load_split("val.txt")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(eval_dataset)} examples")
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train(); trainer.save_model()
    print("✅ Model saved")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--lang", default="rel"); p.add_argument("--add-token", action="store_true")
    p.add_argument("--batch", type=int, default=8); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200,
        help="Save checkpoint every N steps")
    main(p.parse_args())