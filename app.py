from fastai.learner import load_learner
from fastai.vision.core import PILImage

learn_inf = load_learner('./model/export.pkl')
learn_inf.dls.vocab

print(f'[INFO]:First image: msi')
img = PILImage.create('./test-data/msi.jpg')

pred,pred_idx,probs = learn_inf.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

print(f'[INFO]:second image: mss')
img = PILImage.create('./test-data/mss.jpg')

pred,pred_idx,probs = learn_inf.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')




