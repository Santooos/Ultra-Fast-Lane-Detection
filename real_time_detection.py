import torch
import cv2
import numpy as np
import scipy.special
import torchvision.transforms as transforms
from model.model import parsingNet
from utils.common import merge_config
from PIL import Image
from data.constant import culane_row_anchor, tusimple_row_anchor

def main():
    # Initialize configuration and model
    args, cfg = merge_config()

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                     use_aux=False).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # Define the transforms
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if cfg.dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError("Dataset not supported: " + cfg.dataset)


    # Set up video capture
    cap = cv2.VideoCapture(1)  # 0 for the default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #convert the frame to a PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the frame
        frame_tensor = img_transforms(frame_pil).unsqueeze(0).cuda()

        # Inference
        with torch.no_grad():
            out = net(frame_tensor)

        print(out.size())

        # Postprocess and visualize the result
        # This post-processing code is adapted from the provided script.
        # You might need to adjust it based on your specific model output and visualization needs.
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out = out[0].data.cpu().numpy()
        # out = out[:, ::-1, :]
        # print("Shape of out before softmax:", out[0].shape)
        # print("Values in out before softmax:", out[0])

        # prob = scipy.special.softmax(out[0], axis=0)
        out = out[:, ::-1, :]
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        # loc = np.sum(prob * idx.reshape(-1, 1, 1), axis=0)
        loc = np.sum(prob * idx, axis=0)
        out = np.argmax(out, axis=0)
        loc[out == cfg.griding_num] = 0
        out = loc

        # Visualization
        for i in range(out.shape[1]):
            if np.sum(out[:, i] != 0) > 2:
                for k in range(out.shape[0]):
                    if out[k, i] > 0:
                        ppp = (int(out[k, i] * col_sample_w * frame.shape[1] / 800) - 1,
                               int(frame.shape[0] * (row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Real-time Lane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
