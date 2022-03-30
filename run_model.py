from re import I
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json

def predict(example, ner_results):
    ans = ["O"]*len(example)
    for i in ner_results:
        for j in range(int(i["start"]),int(i["end"])):
            if "B" in i["entity"]:
                if j == int(i["start"]):
                    ans[j] = i['entity']
                else:
                    temp = "I-" + i['entity'][2:]
                    ans[j] = temp
            else:
                ans[j] = i['entity']
    
    return ans

def pred_result(token,model,test_file):
    tokenizer = AutoTokenizer.from_pretrained(token)
    model = AutoModelForTokenClassification.from_pretrained(model)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer,ignore_labels=[""])

    word = []
    with open(test_file, newline='') as jsonfile:
        for i in jsonfile:
            obj = json.loads(i)
            temp = obj["tokens"]
            # act += temp
            temp = "".join(str(x) for x in temp)
            word.append(temp)

    result = []
    for i in word:
        ner_results = nlp(i)
        result.append(predict(i, ner_results))
    
    return result

if __name__ == '__main__':
    # load model and tokenizer
    
    token = "/Users/mac/Desktop/eland-intern/final_data/model/best_model_0317_4" #"bert-base-chinese"
    model = "/Users/mac/Desktop/eland-intern/final_data/model/best_model_0317_4"
    test_file = './train_data/test_file.json'

    result = pred_result(token, model, test_file)

    # with open('./predict/prediction_0323_0.txt', 'w') as f:
    #     for line in result:
    #         line = " ".join(str(x) for x in line)
    #         f.write(line)
    #         f.write('\n')