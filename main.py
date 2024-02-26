import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_linear_schedule_with_warmup, BartTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm
import wandb
import pickle
import ujson as json
from utils import format_relationships


class DocREDDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_input_length=1024, max_target_length=1024):
        with open(file_path, 'rb') as file:
            self.dataset = pickle.load(file)

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Preprocess the dataset
        self.preprocessed_data = [self.preprocess_function(item) for item in tqdm(self.dataset)]

    def preprocess_function(self, example):
        source = example['source']
        target = example['target']
        model_inputs = self.tokenizer(source, max_length=self.max_input_length, truncation=True, padding='max_length',
                                      return_tensors="pt")

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target, max_length=self.max_target_length, truncation=True, padding="max_length",
                                    return_tensors="pt")

        model_inputs["labels"] = labels["input_ids"].squeeze(0)
        model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

        return model_inputs

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, tokenizer, device, model_save_path="./best_model_checkpoint",
                 batch_size=4, epochs=20, lr=5e-6):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.best_val_loss = float('inf')

        # Include Warm-up steps to stabilize early training
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_dataset) * epochs
        warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warm-up
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)

        # For Mixed Precision Training
        self.scaler = GradScaler()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            # Enable automatic mixed precision
            with autocast():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

            self.scaler.scale(loss).backward()

            # Gradient clipping
            # Unscale the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip gradients to help prevent the exploding gradient problem
            clip_grad_norm_(self.model.parameters(),
                            max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()  # Update the learning rate
            self.scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output_loss = self.model(**batch)
            loss = output_loss.loss
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        return avg_loss

    def train(self):
        for epoch in range(self.epochs):
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

            avg_train_loss = self.train_epoch(train_dataloader, epoch)
            avg_val_loss = self.eval_epoch(val_dataloader, epoch)

            wandb.log({"train_loss": avg_train_loss, "epoch": epoch})
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

            print(f"Epoch {epoch + 1}: Training Loss {avg_train_loss}, Validation Loss {avg_val_loss}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                print(f"New best validation loss {self.best_val_loss}. Saving model and tokenizer.")
                self.model.save_pretrained(self.model_save_path)
                self.tokenizer.save_pretrained(self.model_save_path)


def main():
    # Arguments
    model_save_path = "./best_model_checkpoint"
    lr = 5e-6
    training_epochs = 40

    wandb.init(project="cs6216",
               name=f"lr{lr}_epoch{training_epochs}"
               #mode="disabled",
               )

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    rel_mapping = json.load(open('./data/raw/docred/rel_info.json', 'r'))

    # Define special tokens
    sep_tokens = ["@REL@", "@NOREL@"]
    rel_tokens = format_relationships(rel_mapping)
    ent_tokens = ["@TIME@", "@ORG@", "@PER@", "@NUM@", "@LOC@", "@MISC@",]

    special_tokens = sep_tokens + rel_tokens + ent_tokens

    special_tokens_dict = {
        'additional_special_tokens': special_tokens
    }

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = DocREDDataset(tokenizer, file_path="./data/processed/docred/train_features.pkl")
    val_dataset = DocREDDataset(tokenizer, file_path="./data/processed/docred/dev_features.pkl")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      val_dataset=val_dataset,
                      tokenizer=tokenizer,
                      device=device,
                      model_save_path=model_save_path,
                      lr=lr,
                      epochs=training_epochs)

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
