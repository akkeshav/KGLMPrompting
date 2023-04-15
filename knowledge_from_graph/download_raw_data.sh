# download ConceptNet
mkdir -p knowledge_from_graph/data/
mkdir -p knowledge_from_graph/data/cpnet/
wget -nc -P knowledge_from_graph/data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd knowledge_from_graphdata/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
# download ConceptNet entity embedding
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../../




# download CommensenseQA dataset
mkdir -p knowledge_from_graph/data/csqa/
wget -nc -P knowledge_from_graph/data/csqa/ https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -nc -P knowledge_from_graph/data/csqa/ https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget -nc -P knowledge_from_graph/data/csqa/ https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl

# create output folders
mkdir -p knowledge_from_graph/data/csqa/grounded/
mkdir -p knowledge_from_graph/data/csqa/graph/
mkdir -p knowledge_from_graph/data/csqa/statement/

