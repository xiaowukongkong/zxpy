import numpy as np
import torch

from algorithm.models.network import Restormer
from algorithm.models.utils import normalization, read_image, inv_normalization, write_image, write_back_dng


def predict(my_cuda, src_path, dest_path, model_path, black_level, white_level):
    print(src_path)
    torch.cuda.set_device(device=my_cuda)
    raw_data_expand_c, src_height, src_width = read_image(src_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)  # (H/2, W/2, 4)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, src_height // 2, src_width // 2, 4), (0, 3, 1, 2))).float()
    # //表示求整，(1, 4, H/2, W/2) torch.Size([1, 4, 1736, 2312])

    net = Restormer()
    if model_path is not None:
        device = torch.device(f"cuda:{my_cuda}" if torch.cuda.is_available() else "cpu")
        model_info = torch.load(model_path, map_location=device)
        net.load_state_dict(model_info['state_dict'])
    net.eval()

    raw_data_expand_c_normal = raw_data_expand_c_normal.cuda(device=my_cuda)
    net.cuda(device=my_cuda)

    with torch.no_grad():
        result_data = net(raw_data_expand_c_normal)

    result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)  # torch.Size([1, 1736, 2312, 4])
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, src_height, src_width)
    write_back_dng(src_path, dest_path, result_write_data)


black_level, white_level = 1024, 16383

model_path = './model.pth'
for i in range(10):
    src_path = f'./testdata/noisy{i}.dng'
    dest_path = f'../data/denoise{i}.dng'
    predict(my_cuda=0,
            src_path=src_path,
            dest_path=dest_path,
            model_path=model_path,
            black_level=black_level,
            white_level=white_level)
