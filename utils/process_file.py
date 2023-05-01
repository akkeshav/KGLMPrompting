import pandas as pd
import random

dataframe = pd.read_json('../data/qasc/knowledge/knowledge_gpt3.dev.qasc.json')

print(dataframe.size)

for index in range(899):
    dataframe.iloc[index]["knowledges"] = []

dataframe.to_json('../data/qasc/knowledge/knowledge_blank.dev.json', orient='records')

# dataframe_1 = pd.read_json('../data/csqa/knowledge/knowledge_1.dev.json')
# complete_data = dataframe_1
# dataframe_2 = pd.read_json('../data/csqa/knowledge/knowledge_gpt3.dev.csqa.json')
# dataframe_3 = pd.read_json('../data/csqa/knowledge/knowledge_para.dev.json')
# dataframe_4 = pd.read_json('../data/csqa/inference/inference_t5-small.knowledge_long_prompt.dev.json')
# dataframe_5 = pd.read_json('../data/csqa/knowledge/knowledge_think_step.dev.json')
#
#
# for index in range(1221):
#     arr1 = random.sample(dataframe_1.iloc[index]["knowledges"], min(4, len(dataframe_1.iloc[index]["knowledges"])))
#     arr2 = random.sample(dataframe_2.iloc[index]["knowledges"], min(4, len(dataframe_2.iloc[index]["knowledges"])))
#     arr3 = random.sample(dataframe_3.iloc[index]["knowledges"], min(4, len(dataframe_3.iloc[index]["knowledges"])))
#     arr4 = random.sample(dataframe_4.iloc[index]["knowledges"], min(4, len(dataframe_4.iloc[index]["knowledges"])))
#     arr5 = random.sample(dataframe_5.iloc[index]["knowledges"], min(4, len(dataframe_5.iloc[index]["knowledges"])))
#     array = arr1 + arr2 + arr3 + arr4 + arr5
#     complete_data.iloc[index]["knowledges"] = array
#
# complete_data.to_json('../data/csqa/knowledge/knowledge_ensemble.json', orient='records')

