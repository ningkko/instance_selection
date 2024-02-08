import os
import torch

import pandas as pd
import tqdm
import random
tqdm.tqdm.pandas()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn

import datetime
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split 

import torch
from sklearn.metrics import f1_score

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans

def flatten(y):
    li = []
    for p in y:
        list_of_lists = [tensor.tolist() for tensor in p]
        li.append(np.array(list_of_lists).T)
    li = np.vstack(li).T
    return li

def calc_metric(y_true, y_pred):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    recall, precision, accuracy = [],[],[]
    for a, b in zip(y_true, y_pred):
        recall.append(recall_score(a, b, average=metric))
        precision.append(precision_score(a, b, average=metric))
        accuracy.append(accuracy_score(a, b))
    return recall, precision, accuracy


def rebuild_df(X, y):
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df = pd.concat([df_X,df_y],axis=1)
    df.columns = ["id","MRN","special_diagnosis","text"] +task_cols
    return df

def log_message(msg):
    with open(f'logs/{measure}.txt', 'a') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{timestamp}] {msg}\n')

class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, task_cols):
        self.tokenizer = tokenizer
        self.texts = df[text_col].values
        self.task_labels = [df[col].values for col in task_cols]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        task_labels = [torch.tensor(label[idx]).long() for label in self.task_labels]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return inputs, task_labels


class MultiTaskDataLoader:
    def __init__(self, df, text_col, task_cols, tokenizer, batch_size, shuffle=True):
        self.dataset = MultiTaskDataset(df, tokenizer, text_col, task_cols)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        for batch in self.dataloader:
            inputs = {key: value.to(device) for key, value in batch[0].items()}
            task_labels = [label.to(device) for label in batch[1]]
            yield inputs, task_labels

    def __len__(self):
        return len(self.dataloader)

    
class MultiTaskModel(torch.nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2, num_classes_task3, num_classes_task4, num_classes_task5, 
                        num_classes_task6=None, num_classes_task7=None, num_classes_task8=None, num_classes_task9=None, num_classes_task10=None):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        # Set requires_grad=True for the last layer parameters
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.1)
        
        if SIDE=="both":
            self.task1_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task1)  # Task 1 output layer
            self.task2_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task2)  # Task 2 output layer
            self.task3_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task3)  # Task 3 output layer
            self.task4_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task4)  # Task 4 output layer
            self.task5_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task5)  # Task 5 output layer
            self.task6_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task1)  # Task 1 output layer
            self.task7_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task2)  # Task 2 output layer
            self.task8_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task3)  # Task 3 output layer
            self.task9_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task4)  # Task 4 output layer
            self.task10_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task5)  # Task 5 output layer
        else:
            self.task1_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task1)  # Task 1 output layer
            self.task2_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task2)  # Task 2 output layer
            self.task3_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task3)  # Task 3 output layer
            self.task4_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task4)  # Task 4 output layer
            self.task5_output = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task5)  # Task 5 output layer

    def forward(self, input_ids, token_type_ids, attention_mask):
        # Pass input through BERT
        outputs=self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        if SIDE=="both":
            task1_logits = self.task1_output(pooled_output)
            task2_logits = self.task2_output(pooled_output)
            task3_logits = self.task3_output(pooled_output)
            task4_logits = self.task4_output(pooled_output)
            task5_logits = self.task5_output(pooled_output)
            task6_logits = self.task6_output(pooled_output)
            task7_logits = self.task7_output(pooled_output)
            task8_logits = self.task8_output(pooled_output)
            task9_logits = self.task9_output(pooled_output)
            task10_logits = self.task10_output(pooled_output)
            return [task1_logits, task2_logits, task3_logits, task4_logits, task5_logits, 
                    task6_logits, task7_logits, task8_logits, task9_logits, task10_logits]


        else:
            task1_logits = self.task1_output(pooled_output)
            task2_logits = self.task2_output(pooled_output)
            task3_logits = self.task3_output(pooled_output)
            task4_logits = self.task4_output(pooled_output)
            task5_logits = self.task5_output(pooled_output)

            return [task1_logits, task2_logits, task3_logits, task4_logits, task5_logits]

def format_scores(scores):
    scores_np = scores.cpu().numpy()
    scores_rounded = np.round(scores_np, 4)
    return scores_rounded.tolist()

def train(model, optimizer, train_loader, val_loader, num_epochs, patience):
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = torch.zeros(1)
    counter = 0
    best_model_state_dict = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch[0]["input_ids"].to(device).squeeze(1)
            token_type_ids = batch[0]["token_type_ids"].to(device).squeeze(1)
            attention_mask = batch[0]["attention_mask"].to(device).squeeze(1)
            task_labels = [label.to(device) for label in batch[1]]
            task_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            task_losses = [criterion(output, labels) for output, labels in zip(task_outputs, task_labels)]
            loss = sum(task_losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        # print('Epoch:', epoch+1, 'Train Loss:', train_loss)

        model.eval()
        task_f1_scores = []
        task_recall_scores = []
        task_precision_scores = []
        task_accuracy_scores = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0]['input_ids'].to(device).squeeze(1)
                token_type_ids = batch[0]["token_type_ids"].to(device).squeeze(1)
                attention_mask = batch[0]['attention_mask'].to(device).squeeze(1)
                task_labels = [label.to(device) for label in batch[1]]
                task_outputs = model(input_ids, token_type_ids, attention_mask)
                task_preds = [torch.argmax(output, dim=1) for output in task_outputs]
                task_f1_scores.append([f1_score(labels.cpu(), preds.cpu(), average=metric)
                                       for labels, preds in zip(task_labels, task_preds)])
                all_labels.append(task_labels)
                all_preds.append(task_preds)
            task_f1_scores = torch.tensor(task_f1_scores).mean(dim=0)
            # print('Validation F1: ', task_f1_scores_rounded)

            val_f1 = task_f1_scores.mean()
            # print('Validation F1 average:', val_f1)
            
            if val_f1 > best_val_f1:
                # print("best model has been updated.")
                # print(f"best val_f1: {best_val_f1}")
                # print(f"new val_f1: {val_f1}")
                
                best_val_f1 = val_f1

                recall, precision, accuracy = calc_metric(all_labels, all_preds) 
                counter = 0
                best_model_state_dict = model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    # print('Validation F1 did not improve for', patience, 'epochs. Training stopped.\n')
                    break

    # print('Validation:', task_f1_scores_rounded)
    # print('Recall:', recall)
    # print('Precision:', precision)
    # print('Accuracy:', accuracy)

    if SIDE=="both":
        best_model = MultiTaskModel(num_classes_task1, num_classes_task2, num_classes_task3, num_classes_task4, num_classes_task5,
                                    num_classes_task6, num_classes_task7, num_classes_task8, num_classes_task9, num_classes_task10)
    else:
        best_model = MultiTaskModel(num_classes_task1, num_classes_task2, num_classes_task3, num_classes_task4, num_classes_task5)
        
    best_model.load_state_dict(best_model_state_dict)
    return best_model, [best_val_f1,task_recall_scores,task_precision_scores,task_accuracy_scores]

def eval_with_preds(model, val_loader, set_name):

    model.eval()
    task_f1_scores = []
    task_f1_scores_macro = []

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0]['input_ids'].to(device).squeeze(1)
            token_type_ids = batch[0]["token_type_ids"].to(device).squeeze(1)
            attention_mask = batch[0]['attention_mask'].to(device).squeeze(1)
            task_labels = [label.to(device) for label in batch[1]]
            task_outputs = model(input_ids, token_type_ids, attention_mask)
            task_preds = [torch.argmax(output, dim=1) for output in task_outputs]
            task_f1_scores.append([f1_score(labels.cpu(), preds.cpu(), average=metric)
                                    for labels, preds in zip(task_labels, task_preds)])
 
            task_f1_scores_macro.append([f1_score(labels.cpu(), preds.cpu(), average="macro")
                                    for labels, preds in zip(task_labels, task_preds)])
            all_labels.append(task_labels)
            all_preds.append(task_preds)
        task_f1_scores = torch.tensor(task_f1_scores).mean(dim=0)
        task_f1_scores_macro = torch.tensor(task_f1_scores_macro).mean(dim=0)
        task_f1_scores_rounded = format_scores(task_f1_scores)
        task_f1_scores_rounded_macro = format_scores(task_f1_scores_macro)
        # print('Validation F1_weighted: ', task_f1_scores_rounded)
        # print('Validation F1_macro: ', task_f1_scores_rounded_macro)

        val_f1 = task_f1_scores.mean()
        val_f1_macro = task_f1_scores_macro.mean()
        # if set_name =="test":
        #     print('Test F1_weighted average:', val_f1.numpy())
        #     print('Test F1_macro average:', val_f1_macro.numpy())
        # else:
        #     print('Validation F1 average:', val_f1.numpy())
        #     print('Validation F1_macro average:', val_f1_macro.numpy())

        best_val_f1 = task_f1_scores_rounded
        best_val_f1_macro = task_f1_scores_rounded_macro


    # print('F1_weighted:', task_f1_scores_rounded)
    # print('F1_macro:', task_f1_scores_rounded_macro)
    # print('Recall:', recall)
    # print('Precision:', precision)
    # print('Accuracy:', accuracy)
    # Return the best model based on validation F1 score
    return [all_labels, all_preds, val_f1.numpy(), best_val_f1, val_f1_macro.numpy(), best_val_f1_macro]

def eval(model, val_loader, set_name):
    [all_labels, all_preds, average_f1_weighted, best_val_f1_weighted, average_f1_macro, best_val_f1_macro] = eval_with_preds(model, val_loader, set_name)
    return [average_f1_weighted, best_val_f1_weighted, average_f1_macro, best_val_f1_macro]


def encode(notes):
    note_reps = []
    for note in notes:
        # Tokenize the note and obtain the input IDs, token type IDs, and attention mask
        encoded_dict = tokenizer.encode_plus(
                            note,
                            add_special_tokens=True,
                            max_length=512,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )

        input_ids = encoded_dict['input_ids'].to(device)
        token_type_ids = encoded_dict['token_type_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            hidden_state = outputs[0][:, 0, :]

        note_reps.append(hidden_state)
    return note_reps

def compute_entropy(probabilities):
    ent = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return ent

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)

from scipy.stats import entropy
from math import ceil

measure = "entropy+cluster"
metric = "weighted"
# instance_num = 100 
SIDE = "both"

text_col = 'text'  # Name of the column that contains the textual input
task1_col = 'left central calyceal dilation'
task2_col = 'left parenchymal appearance abnormal'
task3_col = 'left parenchymal thickness abnormal'
task4_col = 'left peripheral calyceal dilation'
task5_col = 'left ureter abnormal'
task6_col = 'right central calyceal dilation'
task7_col = 'right parenchymal appearance abnormal'
task8_col = 'right parenchymal thickness abnormal'
task9_col = 'right peripheral calyceal dilation'
task10_col = 'right ureter abnormal'


task_cols = [task1_col, task2_col, task3_col, task4_col, task5_col,
             task6_col, task7_col, task8_col, task9_col, task10_col]
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

batch_size = 10
num_epochs = 25
lr = 5e-5
patience = 3

# Initialize the model and optimizer
num_classes_task1 = 3
num_classes_task2 = 3
num_classes_task3 = 3
num_classes_task4 = 3
num_classes_task5 = 3
num_classes_task6 = 3
num_classes_task7 = 3
num_classes_task8 = 3
num_classes_task9 = 3
num_classes_task10 = 3



with open("seed.txt", "r") as file:
    contents = file.read()
    SEEDs = [int(s) for s in contents.split("\n") if s]

n_clusters = 25
for exp_num in [1,2,3,4,5,6]:
    print(f"Experiment {exp_num}")
    exp_folder = f"{measure}/exp{exp_num}/"
    os.makedirs(f"{measure}/", exist_ok=True)
    os.makedirs(exp_folder, exist_ok=True)

    SEED = SEEDs[exp_num]
    set_seed(SEED)

    for iter_num in range(0,14):
        if iter_num in [0,1,2,3,4,5,6]:
            instance_num = 150
        else: 
            instance_num = 250
            
        print(f"Round {iter_num}")
        try:
            data_path = f"{exp_folder}data/iter{iter_num+1}/"
            os.makedirs(data_path, exist_ok=True)

            iter_col = f"iter{iter_num}"
            previous_iter_col = f"iter{iter_num-1}"

            if iter_num!=0:
                df = pd.read_csv(f"{exp_folder}data/train_test_{measure}.csv")
            else:
                df = pd.read_csv(f"data/train_test_v0(3.21).csv")
                X_train, y_train, X_test, y_test = iterative_train_test_split(df[['id', 'MRN', 'special_diagnosis', 'text']].values, df[task_cols].values, test_size = 0.2)
                train_df = rebuild_df(X_train, y_train)
                test_df = rebuild_df(X_test, y_test)
                df["set"] = "unlabled"
                df.set.loc[df.id.isin(test_df.id.tolist())] = "test"

            test_df = df[df.set=="test"]

            if iter_num!=0:
                iter_df = pd.read_csv(f"{exp_folder}data/iter{iter_num}/{measure}.csv")
                train_val_df = df[~df[previous_iter_col].isna()]
                iter_data = pd.concat([train_val_df, iter_df])
            else:
                iter_data = df[df.set!="test"].sample(instance_num, random_state=SEED)
            len(iter_data.id.unique())

            df[iter_col] = None
            df[iter_col].loc[df.id.isin(iter_data.id.tolist())] = True

            iter_cols = [c for c in df.columns if "iter" in c]
            iter_cols

            X_train, y_train, X_test, y_test = iterative_train_test_split(iter_data[['id', 'MRN', 'special_diagnosis', 'text']].values, iter_data[task_cols].values, test_size = 0.2)
            train_df = rebuild_df(X_train, y_train)
            val_df = rebuild_df(X_test, y_test)

            df["set"] = "unlabled"
            df.set.loc[df.id.isin(test_df.id.tolist())] = "test"
            df.set.loc[df.id.isin(train_df.id.tolist())] = "train"
            df.set.loc[df.id.isin(val_df.id.tolist())] = "val"

            df.to_csv(f"{exp_folder}data/train_test_{measure}.csv",index=False)

            val_results, test_results = [], []

            # Create the training and validation data loaders
            train_loader = MultiTaskDataLoader(train_df, text_col, task_cols, tokenizer, batch_size, shuffle=True)
            val_loader = MultiTaskDataLoader(val_df, text_col, task_cols, tokenizer, batch_size, shuffle=False)
            test_loader = MultiTaskDataLoader(test_df, text_col, task_cols, tokenizer, batch_size, shuffle=False)

            model = MultiTaskModel(num_classes_task1, num_classes_task2, num_classes_task3, num_classes_task4, num_classes_task5,
                                num_classes_task6, num_classes_task7, num_classes_task8, num_classes_task9, num_classes_task10)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.to(device)
            print("Training...")
            best_model, best_scores = train(model, optimizer, train_loader, val_loader, num_epochs, patience)
            os.makedirs(f'{exp_folder}/models', exist_ok=True)
            torch.save(best_model.state_dict(), f'{exp_folder}/models/iter{iter_num}_{measure}.pt')

            print("Predicting...")
            [average_f1_weighted, best_val_f1_weighted, average_f1_macro, best_val_f1_macro] = eval(best_model.to(device), val_loader, "validation")
            log_message(f"Experiment {exp_num}, round {iter_num}")
            log_message(f"Validation F1 weighted: {best_val_f1_weighted}; average F1_weighted: {average_f1_weighted}")
            log_message(f"Validation F1 macro: {best_val_f1_macro}; average F1_macro: {average_f1_macro}")
            [average_f1_weighted, best_val_f1_weighted, average_f1_macro, best_val_f1_macro] = eval(best_model.to(device), test_loader, "test")
            log_message(f"Test F1 weighted: {best_val_f1_weighted}; average F1_weighted: {average_f1_weighted}")
            log_message(f"Test F1 macro: {best_val_f1_macro}; average F1_macro: {average_f1_macro}")

            print("Curating new data...")

            outputs = eval_with_preds(model.to(device), val_loader, "val")

            unlabeled_df = df[(df.set != "test") & (df[f"iter{iter_num}"].isna())]
            unlabeled_reps = encode(unlabeled_df.text.tolist())
            unlabeled_emb = torch.cat(unlabeled_reps)
            val_preds = flatten(outputs[1]).T
            val_preds = pd.DataFrame(val_preds, columns=task_cols)
            val_df = val_df.reset_index(drop=True)
            unlabeled_emb_np = unlabeled_emb.cpu().numpy()

            model.to(device)

            text = unlabeled_df["text"].tolist()
            # Tokenize text and create input tensors
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the device

            num_batches = ceil(len(text) / batch_size)

            entropy_list = []

            with torch.no_grad():
                for idx, text_case in enumerate(text):
                    input_case = tokenizer(text_case, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    input_case = {k: v.to(device) for k, v in input_case.items()}  # Move input tensors to the device

                    logits_list = model(input_ids=input_case["input_ids"], token_type_ids=input_case["token_type_ids"], attention_mask=input_case["attention_mask"])

                    entropy_case = []
                    for task_index in range(len(logits_list)):
                        task_logits = logits_list[task_index]
                        probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy().squeeze()
                        ent = compute_entropy(probabilities)
                        entropy_case.append(ent)

                    entropy_list.append(entropy_case)
            entropy_scores = np.vstack(entropy_list)
            score_df = pd.DataFrame(entropy_scores)
            score_df.columns = task_cols
            ## --
            new_series = pd.Series(np.sum(score_df, axis=1), index=score_df.index)
            unlabeled_df = unlabeled_df.reset_index(drop=True)
            unlabeled_df["entropy_score"] = new_series
            # unlabeled_df = unlabeled_df.sort_values("entropy_score", ascending=False)
            # Perform KMeans clustering on the unlabeled embeddings
            n_sample_per_cluster = int(instance_num / n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(unlabeled_emb_np)
            unlabeled_df['cluster'] = kmeans.labels_

            # Find the n_sample_per_cluster cases with the highest entropy in each cluster. If there are not enough cases in a cluster, take all of them. 
            selected_cases = pd.DataFrame()
            for cluster in range(n_clusters):   
                cluster_df = unlabeled_df[unlabeled_df.cluster==cluster]
                cluster_df = cluster_df.sort_values("entropy_score", ascending=False)
                cluster_df = cluster_df.iloc[:n_sample_per_cluster]
                selected_cases = pd.concat([selected_cases, cluster_df])

            # count the number of total cases selected so far. If the total number < instance_num, take the remaining cases with the highest entropy.   
            selected_num = len(selected_cases)
            if selected_num < instance_num:
                remaining_num = instance_num - selected_num
                print(f"Number of remaining cases: {remaining_num}")
                log_message(f"Number of remaining cases: {remaining_num}")

                remaining_cases = unlabeled_df[~unlabeled_df.index.isin(selected_cases.index)]
                remaining_cases = remaining_cases.sort_values('entropy_score', ascending=False)
                remaining_cases = remaining_cases.iloc[:remaining_num]
                selected_cases = pd.concat([selected_cases, remaining_cases])
            else:
                print(f"Number of remaining cases: {0}")
                log_message(f"Number of remaining cases: {0}")

            print(f"selected {len(selected_cases)} new samples" )
            selected_cases.to_csv(f"{exp_folder}data/iter{iter_num+1}/{measure}.csv", index=False)
        except:
            print("failed")
