# advhat_pytorch
advhat: For learning

Task: target attack with cosine_similarity. With a target victim and generate adversial examples to imitate the victims.

The result is:

![Dick_Cheney_Dick_Cheney_0013](https://github.com/YinDFY/advhat_pytorch/assets/127073326/d3f8e70c-63f4-437c-9d22-3505b39b5d49)


When train the adversial examples, I use the famous face recognization model: facenet. And test it on the model: mobile_Face, irse50 and ir152 in a black box attack way.
In order to maintain the scheme's perfomance, I did not modify the raw off plain method and stn net.

Due to limited personal abilities, there may be errors in the project. I hope readers will point out errors so I can correct and learn.

Pretrained Model can be found in https://github.com/TencentYoutuResearch/Adv-Makeup

