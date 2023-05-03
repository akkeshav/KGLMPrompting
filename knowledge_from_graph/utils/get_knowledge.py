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
                    triple = triple.replace("_", " ")
                    if index2conid[index_row] in list_important:
                        dict_final[index2conid[index_row]].append(triple)
                    else:
                        dict_final[index2conid[index_column]].append(triple)

    return {key: (".".join(value))[:400] for key, value in dict_final.items()}


def get_knowledge_statements_relevance_scoring(ground_statement_object):
    concepts_list = ground_statement_object["concepts"]
    adj, cids, adj_original = concepts2adj(concepts_list, pruned_graph, vocab_graph)
    index2conid = {index: id for index, id in enumerate(concepts_list)}
    knowledge_statements_relevance_con = get_importance_based_relevance(ground_statement_object, index2conid,
                                                                        adj_original)

    return knowledge_statements_relevance_con


def get_knowledge_set_relevance_scoring():
    """
    This function returns knowledge statements using Question Context Relevance Scoring in the extracted subgraph
    from Concept Net For extracted topic entities.
    """
    knowledge_set = {}
    for index_statement in tqdm(range(len(df_statement))):
        question_knowledge_set = {}
        for index_statement_option in df_statement.iloc[index_statement]["statements"]:
            statement = index_statement_option["statement"]
            statements_set = {}
            for index_ground in range(len(df_grounded)):
                if df_grounded.iloc[index_ground]["sent"] == statement:
                    statements_set = get_knowledge_statements_relevance_scoring(test_pruned_graph_df[index_ground])
                    break
            question_knowledge_set.update(statements_set)

        question_knowledge_set_array = [value for key, value in question_knowledge_set.items() if value != ''][:20]
        knowledge_set[df_statement.iloc[index_statement]["question"]["stem"]] = question_knowledge_set_array

    return knowledge_set


def are_dest_concepts_visited(concepts, visited):
    """
    This function check whether all the topic entities are visited.
    """
    for concept in concepts:
        if visited[concept] == 0:
            return False
    return True


def dfs(source, visited, adj, path, concepts, relation, index2conid):
    for index, edge in enumerate(adj[source]):
        if edge == 1 and visited[index] == 0 and not are_dest_concepts_visited(concepts, visited):
            visited[index] = 1
            path.append(id2concept[index2conid[source]] + " " + relation + " " + id2concept[index2conid[index]])
            dfs(index, visited, adj, path, concepts, relation, index2conid)


def dfs_reverse(source, visited, adj, path, concepts, relation, index2conid):
    if are_dest_concepts_visited(concepts, visited):
        return True
    for index, edge in enumerate(adj[source]):
        if edge == 1 and visited[index] == 0:
            visited[index] = 1
            if dfs_reverse(index, visited, adj, path, concepts, relation):
                path.append(id2concept[index2conid[source]] + " " + relation + " " + id2concept[index2conid[index]],
                            index2conid)
                return True


def get_knowledge_statements_DFS(ground_statement_object):
    concepts_list = ground_statement_object["concepts"]
    question_list = [index for index, data in enumerate(ground_statement_object["qmask"]) if data == True]
    answer_list = [index for index, data in enumerate(ground_statement_object["amask"]) if data == True]
    destination_list = question_list + answer_list
    adj, cids, adj_original = concepts2adj(concepts_list, pruned_graph, vocab_graph)
    index2conid = {index: id for index, id in enumerate(concepts_list)}

    statements = []
    for index_rel, rel in enumerate(adj_original):
        visited = [0 for ind in range(len(concepts_list))]
        visited[destination_list[0]] = 1
        path = []
        dfs(destination_list[0], visited, rel, path, destination_list, merged_relations_dict[id2relation[index_rel]],
            index2conid)
        statements.append(". ".join(path))

    return statements


def get_knowledge_set_DFS():
    """
    This function returns knowledge statements using Depth first search in the extracted subgraph from Concept Net For
    extracted topic entities.
    """

    knowledge_set = {}
    for index_statement in tqdm(range(len(df_statement))):
        question_knowledge_set = []
        for index_statement_option in df_statement.iloc[index_statement]["statements"]:
            statement = index_statement_option["statement"]
            knowledge_list = []
            for index_ground in range(len(df_grounded)):
                if df_grounded.iloc[index_ground]["sent"] == statement:
                    knowledge_list = knowledge_list + get_knowledge_statements_DFS(test_pruned_graph_df[index_ground])
                    break
            for stat in knowledge_list:
                if stat != '':
                 question_knowledge_set.append(stat)

        knowledge_set[df_statement.iloc[index_statement]["question"]["stem"]] = question_knowledge_set[:20]

    return knowledge_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    load_resources(vocab_graph)
    # Uncomment the type of analysis you want to perform
    knowledge_set = get_knowledge_set_relevance_scoring()
    # knowledge_set = get_knowledge_set_DFS()
    dataframe_dev = pd.read_json(args.input)
    for question, knowledge in knowledge_set.items():
        for index, data in dataframe_dev.iterrows():
            if data["query"] == question:
                dataframe_dev.iloc[index]["knowledges"] = knowledge
                break

    dataframe_dev.to_json(args.output, orient='records')


if __name__ == '__main__':
    main()
