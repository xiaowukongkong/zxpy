1. 参赛者需要根据举办方提供的10张noisy图片提交相应10张denoise图片存放至文件夹“data”下，
命名方式为denoise0.dng至denoise9.dng，
注意上传denoise RAW图值域为[black_level, white_level] = [1024,16383]，可参照baseline代码；
2. 参赛者需要提交模型文件和参数文件至文件夹“algorithm/models/”下，模型文件命名方式为network.py，
参数文件命名pytorch对应model.pth，tensorflow对应model.h5。模型参数文件大小限制为50M；
3. 若使用非AI方法，算法文件提交至以上相同路径，文件命名为alg.py；
4. 参赛者需要提交文档报告阐述所使用方法，文档存放在algorithm二级目录下；
5. data和algorithm按照二级目录结构进行放置，将二级目录放置于命名为result的一级目录内，压缩成.zip格式上传；
