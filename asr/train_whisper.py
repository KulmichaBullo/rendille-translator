#!/usr/bin/env python3
"""Fine-tune Whisper on Rendille audio using LoRA (parameter-efficient)"""
import argparse, json, torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

def main(args):
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Add language token if needed
    if args.add_lang_token:
        processor.tokenizer.add_tokens([f"<|{args.lang}|>"])
        model.resize_token_embeddings(len(processor.tokenizer))

    # Load dataset
    dataset = load_dataset("json", data_files={"train": args.manifest}, split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(example):
        audio = example["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        labels = processor.tokenizer(example["text"]).input_ids
        return {"input_features": inputs.input_features[0], "labels": labels}

    dataset = dataset.map(prepare, remove_columns=dataset.column_names)

    # LoRA config (train only attention heads)
    lora_cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10, save_steps=100,
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=processor.feature_extractor)
    trainer.train()
    trainer.save_model()
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--manifest", default="asr/data/train.jsonl")
    p.add_argument("--output-dir", default="asr/checkpoints/whisper-rel")
    p.add_argument("--lang", default="rendille")
    p.add_argument("--add-lang-token", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--r", type=int, default=16)  # LoRA rank
    p.add_argument("--alpha", type=int, default=32)
    main(p.parse_args())
