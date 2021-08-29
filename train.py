#coding=gbk



import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import data_loader, data_loader_test
from torchvision import transforms as torchtrans
from dataset import dataset_test, plot_img_bbox
from engine import train_one_epoch, evaluate

def get_object_detection_model(num_classes):
    # 加载在COCO上预先训练过的模型（会下载对应的权重）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头替换预先训练好的头
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# to train on gpu if selected.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


num_classes = 4

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
# training for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')




def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')



# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# pick one image from the test set
img, target = dataset_test[5]
# put the model in evaluation mode
with torch.no_grad():
    prediction = model([img.to(device)])[0]


print('predicted #boxes: ', len(prediction['labels']))
print('real #boxes: ', len(target['labels']))

nms_prediction = apply_nms(prediction, iou_thresh=0.2)
print('NMS APPLIED MODEL OUTPUT')
print(torch_to_pil(img))
plot_img_bbox(torch_to_pil(img), nms_prediction)


# pick one image from the test set
img, target = dataset_test[5]

print('EXPECTED OUTPUT')
print(torch_to_pil(img))
plot_img_bbox(torch_to_pil(img), target)


print('MODEL OUTPUT')
plot_img_bbox(torch_to_pil(img), prediction)



nms_prediction = apply_nms(prediction, iou_thresh=0.2)
print('NMS APPLIED MODEL OUTPUT')
plot_img_bbox(torch_to_pil(img), nms_prediction)


