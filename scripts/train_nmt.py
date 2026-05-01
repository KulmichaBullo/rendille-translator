#!/usr/bin/env python3
import argparse, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
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
        save_steps=200, eval_steps=200, logging_steps=50, predict_with_generate=True,
    )
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=None, # TODO: load_dataset("text", data_files={"train": str(DATA/"train.txt")})
        eval_dataset=None,
        tokenizer=tokenizer,
    )
    trainer.train(); trainer.save_model()
    print("✅ Model saved")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--lang", default="rel"); p.add_argument("--add-token", action="store_true")
    p.add_argument("--batch", type=int, default=8); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--epochs", type=int, default=10)
    main(p.parse_args())