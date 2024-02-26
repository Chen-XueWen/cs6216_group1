import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pickle
from tqdm import tqdm


def generate_RE_seq(dataset_path):

     device = torch.device("cuda:2")
     model_save_path = './best_model_checkpoint2_lr5e-6_bkup'
     tokenizer = BartTokenizer.from_pretrained(model_save_path)
     model = BartForConditionalGeneration.from_pretrained(model_save_path)
     model.to(device)

     with open(dataset_path, 'rb') as file:
          dataset = pickle.load(file)

     print("Generating Output")
     for data in tqdm(dataset):
          source = data['source']
          input_ids = tokenizer.encode(source, max_length=1024, truncation=True, padding='max_length',
                                       return_tensors="pt")
          output_ids = model.generate(input_ids.to(device), max_length=1024, num_beams=3, early_stopping=True)
          output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
          data['output'] = output_text

     pickle.dump(dataset, open('dev_processed.pkl', 'wb'))

     return None


def clean_noisy_seq(seq):
     cleaned_string = seq.replace("</s>", "").replace("<s>", "")
     if not cleaned_string.lstrip().startswith('@REL@'):
          cleaned_string = '@REL@ ' + cleaned_string
     if not cleaned_string.rstrip().endswith('@NOREL@'):
          cleaned_string = cleaned_string + ' @NOREL@'

     norel_token_count = cleaned_string.count('@NOREL')

     if norel_token_count == 1:
          # replace the last @REL@ with @NOREL@ based on observation
          last_rel_index = cleaned_string.rfind('@REL@')
          cleaned_string = cleaned_string[:last_rel_index] + '@NOREL@' + cleaned_string[last_rel_index + len('@REL@'):]

     # Remove NBSP
     cleaned_string = cleaned_string.replace('\xa0', ' ')
     return cleaned_string


def parse_information(input_str):
     # Relationships --> Dictionary containing 2 strings (entity pairs) have to break down into sets later
     relationships = {}
     rel_text = input_str.split("@NOREL@")[0].strip()
     rel_list = [item.strip() for item in rel_text.split('@REL@')[1:]]

     for item in rel_list:
          split_item = item.split("@")
          head_set = set([item for item in split_item[0].replace(" ", "").lower().split(';') if item])
          tail_set = set([item for item in split_item[2].replace(" ", "").lower().split(';') if item])
          entity_pair = (frozenset(head_set), frozenset(tail_set))
          rs = split_item[5:]
          rs_cleaned = set([item for item in rs if item.strip()])
          relationships[entity_pair] = rs_cleaned

     norel_text = input_str.split("@NOREL@")[1].strip()

     # Coreferences --> List of Sets
     norel_split = norel_text.split('@')
     norel_ent_list = []
     for i, token in enumerate(norel_split):
          if token in ["TIME", "ORG", "PER", "NUM", "LOC", "MISC",]:
               ent_string = norel_split[i-1].replace(" ", "").lower()
               ent_set = set([item for item in ent_string.split(';') if item])
               norel_ent_list.append(ent_set)

     rel_ent_list = []
     for rel in rel_list:
          rel_split = rel.split('@')
          for i, token in enumerate(rel_split):
               if token in ["TIME", "ORG", "PER", "NUM", "LOC", "MISC",]:
                    ent_string = rel_split[i-1].replace(" ", "").lower()
                    ent_set = set([item for item in ent_string.split(';') if item])
                    rel_ent_list.append(ent_set)
     unique_sets = list(set(frozenset(s) for s in rel_ent_list))
     rel_unique_sets_list = [set(s) for s in unique_sets]

     coreferences = rel_unique_sets_list + norel_ent_list

     # Mentions --> Flattened Coreferences (set)
     mentions = {item for sublist in coreferences for item in sublist}
     return (mentions, coreferences, relationships)

def evaluate_mentions_scores(target_mentions, output_mentions, tp_m, fp_m, fn_m):
     tp_m += len(target_mentions & output_mentions)
     fp_m += len(output_mentions - target_mentions)
     fn_m += len(target_mentions - output_mentions)
     return tp_m, fp_m, fn_m

def evaluate_coreferences_scores(target_corefs, output_corefs, tp_c, fp_c, fn_c):
     target_frozensets = {frozenset(item) for item in target_corefs}
     output_frozensets = {frozenset(item) for item in output_corefs}

     tp_c += len(target_frozensets & output_frozensets)
     fp_c += len(output_frozensets - target_frozensets)
     fn_c += len(target_frozensets - output_frozensets)

     return tp_c, fp_c, fn_c

def evaluate_relationships_scores(target_relationships, output_relationships, tp_r, fp_r, fn_r):

     tp_cnt = 0
     fp_cnt = 0
     target_cnt = 0

     for key in output_relationships:
          head, tail = key
          if (head, tail) in target_relationships or (tail, head) in target_relationships:
               rels = []
               if (tail, head) in target_relationships:
                    for r in output_relationships[key]:
                         # Flip Direction
                         if '_HEAD' in r:
                              rels.append(r.replace('_HEAD', '_TAIL'))
                         else:
                              rels.append(r.replace('_TAIL', '_HEAD'))
                    output_rels_set = set(rels)
                    tp_cnt += len(output_rels_set & target_relationships[(tail, head)])
               else:
                    for r in output_relationships[key]:
                         rels.append(r)
                    output_rels_set = set(rels)
                    tp_cnt += len(output_rels_set & target_relationships[(head, tail)])
                    fp_cnt += len(output_rels_set - target_relationships[(head, tail)])
          else:
               fp_cnt += len(output_relationships[key])

     for key in target_relationships:
          target_cnt += len(target_relationships[key])

     fn_cnt = target_cnt - tp_cnt
     tp_r += tp_cnt
     fp_r += fp_cnt
     fn_r += fn_cnt

     return tp_r, fp_r, fn_r


def calculate_f1(tp, fp, fn):
     precision = tp / (tp + fp)
     recall = tp / (tp + fn)
     f1_score = 2 * (precision * recall) / (precision + recall)

     return precision, recall, f1_score

def main():
     # Run this first to store output so it can be accessed anytime
     #dev_path = './data/processed/docred/dev_features.pkl'
     #generate_RE_seq(dataset_path=dev_path)

     with open('dev_processed.pkl', 'rb') as file:
          dev_dataset = pickle.load(file)

     tp_m, fp_m, fn_m = 0, 0, 0
     tp_c, fp_c, fn_c = 0, 0, 0
     tp_r, fp_r, fn_r = 0, 0, 0

     for i, data in enumerate(tqdm(dev_dataset)):
          source = data['source']
          target = data['target']
          output = data['output']
          label_target = parse_information(target)
          cleaned_output =  clean_noisy_seq(output)
          label_output = parse_information(cleaned_output)
          print("")
          print(label_target)
          print(label_output)
          # Calculate TP, FP and FN for each component
          tp_m, fp_m, fn_m = evaluate_mentions_scores(
               target_mentions=label_target[0],
               output_mentions=label_output[0],
               tp_m=tp_m, fp_m=fp_m, fn_m=fn_m)

          tp_c, fp_c, fn_c = evaluate_coreferences_scores(
               target_corefs=label_target[1],
               output_corefs=label_output[1],
               tp_c=tp_c, fp_c=fp_c, fn_c=fn_c
          )

          tp_r, fp_r, fn_r = evaluate_relationships_scores(
               target_relationships=label_target[2],
               output_relationships=label_output[2],
               tp_r=tp_r, fp_r=fp_r, fn_r=fn_r
          )

     # Mentions F1
     precision_m, recall_m, f1_score_m = calculate_f1(tp_m, fp_m, fn_m)

     # Coreferences F1
     precision_c, recall_c, f1_score_c = calculate_f1(tp_c, fp_c, fn_c)

     # Relationships F1
     precision_r, recall_r, f1_score_r = calculate_f1(tp_r, fp_r, fn_r)

     print(f"Mentions F1: {f1_score_m}, Precision: {precision_m}, Recall: {recall_m}")
     print(f"Coreferences F1: {f1_score_c}, Precision: {precision_c}, Recall: {recall_c}")
     print(f"Relationships F1: {f1_score_r}, Precision: {precision_r}, Recall: {recall_r}")


if __name__ == "__main__":
     main()
