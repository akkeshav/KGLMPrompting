import pandas as pd
from graph import concepts2adj
from conceptnet import merged_relations
from conceptnet import merged_relations_dict
from tqdm import tqdm
import argparse


concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

pruned_graph = 'knowledge_from_graph/data/cpnet/conceptnet.en.pruned.graph'
vocab_graph = 'knowledge_from_graph/data/cpnet/concept.txt'
df_statement = pd.read_json('knowledge_from_graph/data/csqa/statement/dev.statement.jsonl', lines=True)
df_grounded = pd.read_json('knowledge_from_graph/data/csqa/grounded/dev.grounded.jsonl', lines=True)
pickleFile = open("knowledge_from_graph/data/csqa/graph/dev.graph.adj.pk", "rb")
test_pruned_graph_df = pd.read_pickle(pickleFile)


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def get_importance_based_relevance(ground_statement_object, index2conid, adj_original):
    list_important = list(ground_statement_object["cid2score"].keys())[:10]
    dict_final = {conId: [] for conId in list_important}
    for index_rel, rel in enumerate(adj_original):
        for index_row, row in enumerate(rel):
            for index_column, edge in enumerate(row):
                if edge == 1 and (
                        index2conid[index_row] in list_important or index2conid[index_column] in list_important):
                    relation_text = merged_relations_dict[id2relation[index_rel]]
                    triple = "{} {} {}".format(id2concept[index2conid[index_row]],
                                               relation_text, id2concept[index2conid[index_column]])
                    if index2conid[index_row] in list_important:
                        dict_final[index2conid[index_row]].append(triple)
                    else:
                        dict_final[index2conid[index_column]].append(triple)

    dict_final_con = {key: (".".join(value))[:400] for key, value in dict_final.items()}
    knowledge_statements = [value for key, value in dict_final_con.items() if value != ''][:4]
    return knowledge_statements


def get_knowledge_statements(ground_statement_object):
    concepts_list = ground_statement_object["concepts"]
    adj, cids, adj_original = concepts2adj(concepts_list, pruned_graph, vocab_graph)
    index2conid = {index: id for index, id in enumerate(concepts_list)}
    knowledge_statements_relevance = get_importance_based_relevance(ground_statement_object, index2conid, adj_original)

    return knowledge_statements_relevance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    load_resources(vocab_graph)
    knowledge_set = {}
    for index_statement in tqdm(range(len(df_grounded))):
        question_knowledge_set = []
        for index_statement_option in df_statement.iloc[index_statement]["statements"]:
            statement = index_statement_option["statement"]
            statements_set = []
            for index_ground in range(len(df_grounded)):
                if df_grounded.iloc[index_ground]["sent"] == statement:
                    statements_set = get_knowledge_statements(test_pruned_graph_df[index_ground])
                    break
            for stat in statements_set:
                question_knowledge_set.append(stat)

        knowledge_set[df_statement.iloc[index_statement]["question"]["stem"]] = question_knowledge_set

    dataframe_dev = pd.read_json(args.input)
    for question, knowledge in knowledge_set.items():
        for index, data in dataframe_dev.iterrows():
            if data["query"] == question:
                dataframe_dev.iloc[index]["knowledges"] = knowledge
                break

    dataframe_dev.to_json(args.output, orient='records')


if __name__ == '__main__':
    main()
