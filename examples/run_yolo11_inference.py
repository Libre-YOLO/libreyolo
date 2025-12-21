from libreyolo import LIBREYOLO 

model = LIBREYOLO(model_path="../weights/libreyolo11n.pt", size="n")

results = model(image="../media/test_image_1_creative_commons.jpg", save=True)

print(results)