from fastai.learner import load_learner
from fastai.vision.core import PILImage

model_inf = load_learner('./model/export.pkl')

img = PILImage.create('./test-data/blk-AACEIDCGKIES-TCGA-AG-3881-01Z-00-DX1.jpg')
what, _, probs = model_inf.predict(img)

print(f"001")

print(f"What is this?: {what}.")
print(f"Probability it's a {what}: {probs[1].item():.6f}")

print()

img = PILImage.create('./test-data/ blk-AACQCATSTDFD-TCGA-CM-4746-01Z-00-DX1.jpg')
what, _, probs = model_inf.predict(img)

print(f"002")
print(f"What is this?: {what}.")
print(f"Probability it's a {what}: {probs[1].item():.6f}")
