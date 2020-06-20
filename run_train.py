#@title Net2 w { form-width: "10%" }
# print("Height:", model.output_height)
train(model,
    train_images =  "/content/drive/My Drive/Colab Notebooks/64by64/train_Exp_1/images_Exp_1",
    train_annotations = "/content/drive/My Drive/Colab Notebooks/64by64/train_Exp_1/manuals_Exp_1",
    checkpoints_path = "/content/drive/My Drive/Colab Notebooks/64by64/checkpoints/vgg_unet_1" , epochs=5
)
