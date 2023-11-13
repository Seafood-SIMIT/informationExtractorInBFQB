import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFn(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5):
        super(LossFn, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, predict, label_conf,label_anchor,label_relations):
        # predictions: 模型的输出 (batch_size, S, S, B * (5 + C))
        # targets: 真实标签 (batch_size, S, S, 5 + C)

        # 提取预测中的位置信息和置信度信息

        # 计算位置损失
        loc_loss = self.location_loss(predict['predict_anchor'], label_anchor)

        # 计算置信度损失
        conf_loss = self.confidence_loss(predict['predict_conf'], label_conf)

        # 计算分类损失
        class_loss = F.cross_entropy(predict['predict_relation'].view(-1, predict['predict_relation'].size(2)), label_relations.view(-1).long())

        # 总体损失
        total_loss = self.lambda_coord * loc_loss + self.lambda_obj * conf_loss + class_loss
        #total_loss = self.lambda_obj * conf_loss 
        #total_loss = class_loss

        return total_loss

    def location_loss(self, pred_boxes, true_boxes):
        # 计算位置损失，使用平方误差损失
        loss = F.mse_loss(pred_boxes, true_boxes, reduction='sum')
        return loss

    def confidence_loss(self, pred_confidence, true_confidence):
        loss = F.mse_loss(pred_confidence, true_confidence, reduction='mean')
        return loss
        # 计算置信度损失，使用二元交叉熵损失
        obj_mask = true_confidence.bool()  # 有目标的网格
        noobj_mask = ~obj_mask  # 无目标的网格

        obj_loss = F.binary_cross_entropy_with_logits(pred_confidence[obj_mask], true_confidence[obj_mask], reduction='sum')
        noobj_loss = F.binary_cross_entropy_with_logits(pred_confidence[noobj_mask], true_confidence[noobj_mask], reduction='sum')

        # 加权和
        loss = self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss
        return loss

