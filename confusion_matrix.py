# Importing the dependancies
from sklearn import metrics
import json
from trainset_prep import get_index_positions
from run_model import*

_path = "0329_1"

_label = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG",
         "B-GORG", "I-GORG", "B-PER", "I-PER"]
_label_sp = ["LOC", "ORG", "GORG", "PER"]

def check_accuracy(pre, act):
    global entry

    data_len = len(pre)
    count = 0
    for i in range(len(pre)):
        temp = act[i]
        ind = pre[i]
        entry[temp][ind] += 1
        if pre[i] == act[i]:
            count += 1

    return count/data_len

def list_split(data):
    title = []
    temp = []
    for i in range(len(data)):
        if i == 0:
            temp.append(data[i])
        elif data[i] == data[i-1] + 1:
            temp.append(data[i])
        else:
            title.append(temp)
            temp = []
        if i == len(data)-1:
            title.append(temp)
    return title

def check(pre, act):
    pre = set(tuple(l) for l in pre)
    act = set(tuple(l) for l in act)
    tp = len(pre & act)
    fp = len(pre - act)
    fn = len(act - pre)

    return tp, fp, fn

def pre_work(pre, act, ner_type):
    b = "B-" + ner_type.upper()
    i = "I-" + ner_type.upper()

    pre_entry = []
    pre_entry += (get_index_positions(pre, b))
    pre_entry += (get_index_positions(pre, i))
    pre_entry.sort()
    pre_entry = list_split(pre_entry)

    act_entry = []
    act_entry += (get_index_positions(act, b))
    act_entry += (get_index_positions(act, i))
    act_entry.sort()
    act_entry = list_split(act_entry)

    return pre_entry, act_entry

def save_json_indent(data,path,way):
    with open(path, way , encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Precision = tp/(tp+fp)，即陽性的樣本中有幾個是預測正確的。
# Recall = tp/(tp+fn)，即事實為真的樣本中有幾個是預測正確的。
def precision_recall(pred_data,act_data):
    precision = {"precision":"precision data"}
    recall = {"recall":"recall data"}

    for j in _label_sp:
        all_tp = 0
        all_fp = 0
        all_fn = 0
        for i in range(len(pred_data)):
            pre, act = pre_work(pred_data[i], act_data[i], j)
            tp, fp, fn = check(pre, act)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        precision_temp = all_tp/(all_tp+all_fp)
        recall_temp = all_tp/(all_tp+all_fn)
        precision[j] = precision_temp
        recall[j] = recall_temp

    recall_row = []

    for i in range(len(pred_data)):
        all_tp = 0
        all_fp = 0
        all_fn = 0
        for j in _label_sp:
            # print(pred_data[i])
            pre, act = pre_work(pred_data[i], act_data[i], j)
            tp, fp, fn = check(pre, act)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        recall_temp = all_tp/(all_tp+all_fn)
        recall_row.append((i,recall_temp))

    # sorted_by_second = sorted(recall_row, key=lambda tup: tup[1])

    return precision,recall


def main():
 # Predicted values
    token = "bert-base-chinese"
    model = "./model/best_model_0323_2"
    test_file = './train_data/test_file.json'

    raw = pred_result(token,model,test_file)

    pred = []
    pred_data = []
                
    for line in raw:
        line = " ".join(str(x) for x in line)
        temp = line.strip("\n").split(" ")
        pred += temp
        pred_data.append(temp)

    # Actual values
    act = []
    act_data = []
    token_data = []
    with open('./train_data/test_file_0323.json', newline='') as jsonfile:
        for i in jsonfile:
            obj = json.loads(i)
            temp = obj["ner_tags"]
            token = obj["tokens"]
            act += temp
            act_data.append(temp)
            token_data.append(token)

    result = metrics.classification_report(act, pred, labels=_label)
    # save confusion matrix
    outfile = open("analysis_result/confusion_matrix_{}.txt".format(_path),"w")
    outfile.write(result)


    total_num = {}
    for i in range(len(_label)):
        total_num[_label[i]] = act.count(_label[i])
    
    entry = {}
    for i in range(len(_label)):
        entry[_label[i]] = {'O': 0, 'B-LOC': 0, 'I-LOC': 0, 'B-ORG': 0,
                        'I-ORG': 0, 'B-GORG': 0, 'I-GORG': 0, 'B-PER': 0, 'I-PER': 0}
    similar = []
    for i in range(len(pred_data)):
        similar.append((i, check_accuracy(pred_data[i], act_data[i])))
    # save 各類別準確度
    for key, values in entry.items():
        for j in values:
            values[j] = values[j]/total_num[j]
    
    precision, recall = precision_recall(pred_data,act_data)

    save_json_indent(entry,'analysis_result/results_{}.json'.format(_path),'w')
    save_json_indent(precision,'analysis_result/results_{}.json'.format(_path),'a')
    save_json_indent(recall,'analysis_result/results_{}.json'.format(_path),'a')

if __name__ == '__main__':
    main()