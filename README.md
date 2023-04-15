# Investigation of joint reasoning with Language models and Knowledge graphs for Commonsense question answering


## Demo video
[Please click to checkout demo video](demo.mp4)

## Usage
### Dependencies
Run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n KG_LM python=3.7
source activate KG_LM
pip install torch==1.8.0
pip install transformers==4.9.1
pip install nltk spacy==2.1.6
python -m spacy download en

# for torch-geometric
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
```


### Download data
We use the question answering datasets (*CommonsenseQA*) and the ConceptNet knowledge graph.
Download all the raw data by
```
./knowledge_from_graph/download_raw_data.sh
```

Preprocess the raw data by running
```
python ./knowledge_from_graph/preprocess.py -p <num_processes>
```
The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair


The resulting file structure will look like:

```plain
.
├── README.md
├── knowledge_from_graph
    ├── data/
        ├── cpnet/                 (prerocessed ConceptNet)
        ├── csqa/
            ├── train_rand_split.jsonl
            ├── dev_rand_split.jsonl
            ├── test_rand_split_no_answers.jsonl
            ├── statement/             (converted statements)
            ├── grounded/              (grounded entities)
            ├── graphs/                (extracted subgraphs)
            ├── ..
```

We can also download preprocessed data using following link

```commandline
https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
```

Name this downloaded folder data and put this folder under knowledge_from_graph folder.

### Get Knowledge statements
By using this file we can get all the knowledge statements for every question present in the dataset.

First, download the dataset for the following tasks:  [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)

Next, use `standardize/csqa_standardize.py` to put the data in a unified format.

Following command requires two required parameters: input and output
- Here, input denotes path of the input file. Similarly, output represents output path of file for getting knowledge 
statements.

```commandline
python ./knowledge_from_graph/utils/get_knowledge.py --input ./data/csqa/knowledge/knowledge_gpt3.dev.csqa.json --output ./data/csqa/knowledge/concept_net.dev.csqa.json
```
### Inference
Once all the required knowledge is retrieved from the KG. We can use Language models like T5 to perform the inference
predict the final answer.

Please use the below-mentioned command for doing inference for knowledge statements generated using ConceptNet. We can change
the input-path parameter to do the inference for other types of knowledge generations such GPT-3 prompt knowledge, random statements, etc.

```commandline
python ./inference/infer_t5.py --task csqa  --model-type t5-small --input-path ./data/csqa/knowledge/concept_net.dev.csqa.json
```

## Acknowledgment
This repo is built upon the following works:

- More precisely, The graph extraction and pruning part is mostly inspired from following work.
```
QA-GNN: Question Answering using Language Models and Knowledge Graphs. https://github.com/michiyasunaga/qagnn
```
- Similarly, prompting and inference module is taken from following work.
```
Generated Knowledge Prompting. https://github.com/liujch1998/GKP
```

