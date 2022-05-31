import os

import numpy as np
import rawpy
import torch


# 作0-1之间的标准化，减最小值除以（最大值-最小值）
def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


# 逆标准化
def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def read_image(input_path):
    raw = rawpy.imread(input_path)  # 'rawpy._rawpy.RawPy'
    raw_data = raw.raw_image_visible  # 返回 ndarray, (H, W)
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)  # 扩展维度为 (H, W, 1)

    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)  # 得到(H/2, W/2, 4)，保留了每一像素的值
    return raw_data_expand_c, height, width  # 返回了原始图像的 height ，width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    # print(input_data.shape,height,width)

    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data  # read_image的逆变换


def write_image_without_batch(input_data, height, width):
    # print(input_data.shape, height, width)
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[:, :, 2 * channel_y + channel_x]
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def save_model(model, loss, psnr, ssim, launchTimestamp):
    print('save models...')
    # print(launchTimestamp)
    torch.save({'state_dict': model.state_dict(), 'best_loss': loss, 'psnr': psnr, 'ssim': ssim},
               './m-' + str(int(launchTimestamp)) + '-' + str("%.4f" % psnr) + ',' + str("%.4f" % ssim) + '.pth')
    print('models has saved...')


def load_model(model, path):
    if path != None:
        model_CKPT = torch.load(path)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
    return model


def transform(x, factor):
    # torch.Size([1, 4, 1736, 2312]) -> torch.Size([1 * factor**2, 4, 1736/factor, 2312/factor])
    [B, C, H, W] = x.shape
    assert H % factor == 0 and W % factor == 0
    ans = torch.zeros((B * factor * factor, C, H // factor, W // factor))
    for row in range(factor):
        for col in range(factor):
            ans[row * factor + col, :, :, :] = x[:, :, row::factor, col::factor]
    return ans


def inv_transform(x, factor):
    [B, C, H, W] = x.shape
    assert B % (factor * factor) == 0
    ans = torch.zeros((B // (factor * factor), C, H * factor, factor * W))
    for row in range(factor):
        for col in range(factor):
            ans[:, :, row::factor, col::factor] = x[row * factor + col, :, :, :]
    return ans
