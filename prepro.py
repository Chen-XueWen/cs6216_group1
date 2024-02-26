from tqdm import tqdm
import ujson as json
import pickle


def read_docred(file_in):
    i_line = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        entities_unique = {}
        entities_str = []
        rels_dict = {}
        ent_with_rel = []
        
        target = []
        source = [item for sublist in sample['sents'] for item in sublist]
        
        if "labels" in sample:
            for i, entity in enumerate(sample['vertexSet']):
                entities_unique[i] = {}
                entities_unique[i]['mentions'] = []
                entities_unique[i]['type'] = entity[0]['type'] #take first mention as the entity type
                for mention in entity:
                    if mention['name'] not in entities_unique[i]['mentions'] :
                        entities_unique[i]['mentions'].append(mention['name'])

            for key in entities_unique:
                mentions_str = "; ".join(entities_unique[key]['mentions'])
                entity_type = entities_unique[key]['type']
                entities_str.append(f"{mentions_str} @{entity_type}@")

            for rel in sample['labels']:
                ent_with_rel.append(rel['h'])
                ent_with_rel.append(rel['t'])

                key = tuple(sorted((rel['h'], rel['t'])))
                if rel['h'] == key[0]:
                    relationship = rel_mapping[rel['r']].upper().replace(" ", "_") + '_HEAD'
                else:
                    relationship = rel_mapping[rel['r']].upper().replace(" ", "_") + '_TAIL'

                if key not in rels_dict:
                    rels_dict[key] = [relationship]
                else:
                    rels_dict[key].append(relationship)

            # Sort relationships alphabetically
            for key, value in rels_dict.items():
                rels_dict[key] = sorted(value)

            # Sort Generation based on entity position
            # DocRED VertexSet is already sorted based on position
            sorted_key = sorted(rels_dict.keys())
            for key in sorted_key:
                target_str = f"@REL@ {entities_str[key[0]]} {entities_str[key[1]]}"
                for rel in rels_dict[key]:
                    target_str += f" @{rel}@"
                target.append(target_str)

            ### For Entity without Relationship ###
            ent_wo_rel = set(entities_unique.keys()).difference(set(ent_with_rel))
            target.append("@NOREL@")
            for entity in sorted(ent_wo_rel):
                target.append(entities_str[entity])
            target.append("@NOREL@")
            target = " ".join(target)
        
        source = " ".join(source)
        i_line += 1
        feature = {
                   'source': source,
                   'target': target,
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    return features


def main():
    global rel_mapping

    rel_mapping = json.load(open('./data/raw/docred/rel_info.json', 'r'))
    
    #train_file = "./data/raw/docred/train_annotated.json"
    dev_file = "./data/raw/docred/dev.json"
    #test_file = "./data/raw/docred/test.json"
    
    #train_features = read_docred(train_file)
    dev_features = read_docred(dev_file)
    #test_features = read_docred(test_file)

    train_out = './data/processed/docred/train_features.pkl'
    dev_out = './data/processed/docred/dev_features.pkl'
    test_out = './data/processed/docred/test_features.pkl'

    outfile = open(train_out, 'wb')
    pickle.dump(train_features, outfile)
    outfile.close()

    outfile = open(dev_out, 'wb')
    pickle.dump(dev_features, outfile)
    outfile.close()

    outfile = open(test_out, 'wb')
    pickle.dump(test_features, outfile)
    outfile.close()


if __name__ == "__main__":
    main()
