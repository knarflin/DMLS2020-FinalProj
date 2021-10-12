- Full data classification
- Divide dataset into 4 sites.
- Get a fairly-good neural architecture.
- Train at an aggregated mode.


- Decide the image preprocessing method
  - crop?
  - rotate?
  - normalized?

readfile


multiclass classification

x = [num_of_files, w, h, channel]
y = [num_of_files]


定義好 cnn input 形式 (h, w, channel) 後
- 前半 Data preprocessing 要符合 cnn 作調整
  - Read image
  - Torch vision
- 後半 CNN Training, Validation, Testing 都要也符合

  model (Classifier
  loss (CrossEntropyLoss
  optimizer (SGD

  random split 將 dataset 分開為 training set 和 testing set
  dataset type 如何只取 img 不取 label 作 val set accuracy

  split data into val + 4 train_site

  docker + mount training data 的靈活控制

  起 docker 跑 process

  !! 為什麼 site 1~3 print 就可以顯示在 logging [N STDOUT] 上？好神阿！

  docker 和 nvidia GPU 的使用，4 個 docker container 共用一個 GPU ?


pytorch
  float32 -> ADAM 
  float16 -> SGD
  float16 -> ADAM (bug)

What is docker volume? 和 docker mount disk 是什麼關係？

#### Bug 
- TypeError: conv2d() got multiple values for keyword argument 'groups
  - 可能是因為 把 torch.tensor 轉成 cuda 餵進 cuda 的 model 裡。
  - /home/r06922149/testzone/crypten-test/CrypTen/crypten/cuda/cuda_tensor.py:204 
  - `x_enc_span, y_enc_span, *args, **kwargs, groups=nb2`
  - keyword argument, positional argument
  - Variable number of arguments
    - `*args` (Non Keyword Arguments)
    - `**kwargs` (Keyword Arguments)




torch.nn.functional.[op] vs torch.nn.[op]
torch.nn.functional.conv2d vs torch.nn.conv2d

#### TODO
- 製作 patch (如何 git branch, commit, pull request)
- docker 內的 crypten 上 patch，然後重製 image ，重產生 container
- 確保 crypten-test.py 的 CNN 可以用到我目前用的那些 Layer。
  - crypten-test.py 的定位是 測試 單個 rank 、測試 torch 和 crypten 共同的部份、測試 torch.nn 的 Layer 是否可以加更多進來
- 確保 crypten 版的 training code 裡 Classifier 可以跑，並且可以使用 GPU 加快速度
- 收尾，製作圖表資料，製作 1/4 投影片 (7-8 min)
- 1/4 中午前提交投影片到 NTU Cool / Grade Scope / Ceiba ?

- 是否可以把 BatchNorm2D 加上去



- 把 mpc_cnn.py (單機、單 thread 的 encrypted training 做好)
  - loss, optimizer,
  - one hot?
  - print loss, print train acc, print val accuracy
  - simple CNN layers -> complex
  - no .gpu() -> with .gpu()



目前把 data leave on per stream 剩下 1307 張 有 label 的
20% validation
80% training -> 4 sites

SGD( lr = 0.01, momentum=0.9 )
  site0 大概只能 fit 到 val acc = 0.22~0.26 (100 epoch 內)
  all 大概可以 fit 到 val acc = 0.54~0.59 (100 epoch 內)


目前在 Xeon Phi (172.16.179.26) 上有測試，但 NNPack 無法支援該硬體，估計是 CPU 不合。

