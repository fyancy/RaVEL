import torch
import torch.nn.functional as F

# https://github.com/Sun-DongYang/Pytorch/blob/master/multiLabel/multiLabel.py


# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为该样本中含有这类标签
def calculate_accuracy_mode_one(model_pred: torch.Tensor, labels: torch.Tensor,
                                shuffled=True, num_cls=8, num_cls_train=8):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确

    # if model_pred.ndim == 1:  % DO NOT USE THIS! Many ZEROS would increase the acc-ml.
    #     model_pred = F.one_hot(model_pred, num_cls)
    #     labels = F.one_hot(labels, num_cls)
    # print("labels:", labels.shape, "pred_probs:", model_pred.shape)

    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()

    _strict_acc = torch.empty(len(labels), device=labels.device)
    for i in range(len(pred_result)):
        _strict_acc[i] = (pred_result[i] == labels[i]).all()
    strict_acc = _strict_acc.mean()

    # if not shuffled and num_cls > num_cls_train:
    #     num_x_per_cls = len(labels) // num_cls
    #     unseen_cls_acc = _strict_acc[num_x_per_cls * num_cls_train:].mean().item()
    #     seen_cls_acc = _strict_acc[:num_x_per_cls * num_cls_train].mean().item()
    #     print(f"SEEN acc.: {seen_cls_acc:.2%}, UNSEEN acc.: {unseen_cls_acc:.2%}")

    pred_one_num = torch.sum(pred_result)  # TP+FP
    target_one_num = torch.sum(labels)  # TP + FN
    true_positive = torch.sum(pred_result * labels)  # TP
    # 模型预测的结果中有多少个是正确的
    precision = true_positive / (pred_one_num+1e-8)
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_positive / (target_one_num+1e-8)

    pred_result, labels = 1-pred_result, 1-labels
    target_zero_num = torch.sum(labels)  # TN + FP
    true_negative = torch.sum(pred_result * labels)  # TN
    acc = (true_positive+true_negative)/(target_one_num+target_zero_num)  # ml-acc

    return precision.item(), recall.item(), acc.item(), strict_acc.item()


# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_accuracy_mode_two(model_pred, labels):
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall


if __name__ == "__main__":
    # pred_probs1 = torch.tensor([[0, 1, 0.3, 0.8, 0.4, 0.6], [0.8, 0.91, 0.5, 0.8, 0.24, 0.86]])
    # labels1 = torch.tensor([[0, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1]])
    pred_probs1 = torch.tensor([0, 2, 3, 4, 5, 3, 0, 1])
    labels1 = torch.tensor([0, 1, 5, 4, 0, 3, 2, 1])
    print(calculate_accuracy_mode_one(pred_probs1, labels1))

    # from sklearn.metrics import accuracy_score
    # accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
