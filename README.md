This is a collection of models and training for taking an image and predicting what US state it was from. FastAI was used for most model development. The training logs can be found at https://wandb.ai/ben_z/geolocation/?nw=nwuserben_z:

![image](https://github.com/bizy-coder/ViT_Geolocation/assets/52185831/4660d1e6-37b1-4e6e-af81-cb77c9b9fc7c)


The accuracy is quite high. The initial model achieved 60% accuracy, with significantly better top-k accuracy. Later models pushed this up to 67%:

![image](https://github.com/bizy-coder/ViT_Geolocation/assets/52185831/45ed1da9-ab7a-4072-ab59-6919680726e3)


Finally, for those interested, here is a chart of what states were easy and hard to identify:
![image](https://github.com/bizy-coder/ViT_Geolocation/assets/52185831/5e9ce5d1-7379-4968-bcf5-81e0475993ff)
