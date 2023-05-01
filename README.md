# Is prompting a better knowledge source than knowledge graphs for answering commonsense questions?


## Knowledge Generation 
### Knowledge Graph Raw data
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
As part of this research work, We only use the CommonsenseQA data for this type of knowledge generation.
We downloaded preprocessed data for our experiments using following link

```commandline
https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
```
After downloading, name the downloaded folder as "data" and put this folder under knowledge_from_graph folder.
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

As an alternative, we can also download raw data and process them. 
Below-mentioned script downloads the raw data.
```
./knowledge_from_graph/download_raw_data.sh
```
Preprocess the raw data by running
```
python ./knowledge_from_graph/preprocess.py -p <num_processes>
```
To summarize the script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair


### Prompting Raw data

First, download the dataset for the following tasks:  [NumerSense](https://github.com/INK-USC/NumerSense), [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa), and [QASC](https://allenai.org/data/qasc).

Next, use `standardize/{task_name}_standardize.py` to put the data in a unified format.


### Knowledge statement Generation
Please use following commands to generate knowledge statements

#### Knowledge Graph
We need two mandatory parameters for following command: input and output
- Here, input denotes path of the input file. Similarly, output represents output path of file for getting knowledge 
statements.

```commandline
python ./knowledge_from_graph/utils/get_knowledge.py --input ./data/csqa/knowledge/knowledge_gpt3.dev.csqa.json --output ./data/csqa/knowledge/concept_net.dev.csqa.json
```

#### Prompting
Use `gpt3_generate_knowledge.py` to generate knowledge for a task dataset.
For example, to generate knowledge for the validation set of NumerSense, run
```
python knowledge/gpt3_generate_knowledge.py \
    --task numersense \
    --input_path data/numersense/validation.json \
    --output_path data/numersense/knowledge/knowledge_gpt3.validation.json \
    --prompt_path knowledge/prompts/numersense_prompt.txt
```

Please find all the prompts in the prompts folder.

### Inference
Once all the required knowledge is retrieved using Knowledge graph and Prompting. We can use Language models like T5-small to perform the inference
predict the final answer.

Please use the below-mentioned command for doing inference for knowledge statements. We can change
the input-path parameter to do the inference for other types of knowledge generations such Zero-shot COT, random knowledge, etc.

```commandline
python ./inference/infer_t5.py --task csqa  --model-type t5-small --input-path ./data/csqa/knowledge/concept_net.dev.csqa.json
```

Similarly, for Numersense we need to use following command:-
```
python inference/infer_numersense_t5.py \
    --model-type t5-11b \
    --input-path data/numersense/knowledge/knowledge_gpt3.validation.json
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

