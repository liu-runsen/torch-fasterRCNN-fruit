'''
@Author：Runsen
'''

PATH = "fasterrcnn_resnet50_fpn.pth"
model = model.load_state_dict(torch.load(PATH))