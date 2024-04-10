# advhat_pytorch
advhat: For learning

Task: target attack with cosine_similarity. With a target victim and generate adversial examples to imitate the victims.

The result is:

![1-ori](https://github.com/YinDFY/advhat_pytorch/assets/127073326/4586591f-f994-4a5f-9c3e-30d3c0cd7fe7)



When train the adversial examples, I use the famous face recognization model: facenet. And test it on the model: mobile_Face, irse50 and ir152 in a black box attack way.
In order to maintain the scheme's perfomance, I did not modify the raw off plain method and stn net.

Due to limited personal abilities, there may be errors in the project. I hope readers will point out errors so I can correct and learn.

Pretrained Model can be found in https://github.com/TencentYoutuResearch/Adv-Makeup

