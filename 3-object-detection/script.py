import torch
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio



def detect (frame, net, transform):
    height, width = frame.shape[:2] # ALSO WORKS AS 0, 1
    transformed_frame = transform(frame)[0]

    x = torch.from_numpy(transformed_frame).permute(2, 0, 1)
    x = x.unsqueeze(0)

    with torch.no_grad():
        y = net(x)

    """ detections = [
            batch,
            number of classes,
            number of occurences,
            (threshold, x0, y0, x1, y1)
        ]
    """
    detections = y.data
    scale = torch.Tensor([width, height, width, height])


    for i in range(detections.size(1)):
        j = 0

        while detections[0, i, j, 0] >= 0.6:
            # vectors = [x0, y0, x1, y1]
            vectors = (detections[0, i, j, 1:] * scale).numpy()

            cv2.rectangle(frame, (int(vectors[0]), int(vectors[1])), (int(vectors[2]), int(vectors[3])), (255, 0, 0), 2)

            cv2.putText(frame, labelmap[i - 1], (int(vectors[0]), int(vectors[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            j += 1

    return frame



neural_network = build_ssd('test')
neural_network.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location= lambda storage, loc: storage))

transform = BaseTransform(neural_network.size, (104.0/256.0, 117.0/256.0, 123.0/256.0))

reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, neural_network.eval(), transform)
    writer.append_data(frame)
    print(i)

writer.close()
