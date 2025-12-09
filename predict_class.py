from torchvision import io as tvio
from torchvision import models
import torchinfo

#画像の読み込み
input_image = tvio.decode_image("assets/IMG_3436.JPG")
print(type(input_image))
print(input_image.shape, input_image.dtype)

# 学習済みモデルの重みを読み込む
weights = models.AlexNet_Weights.DEFAULT
#weights = models.VGG19_BN_Weights.DEFAULT

# モデルを作る
model = models.alexnet(weights=weights)
#print(model)
torchinfo.summary(model)

#入力画像の前処理を行う
preprocess = weights.transforms()

#パッチにする
#(C,H,W) ->(1,C,H,W)
batch = preprocess(input_image).unsqueeze(dim=0)
print(batch.shape)

# モデルの推論モードにする
model.eval()

#バッチに対して推論(モデルの計算)を行う
output_logits = model(batch)
print(output_logits.shape, output_logits.dtype)

#バッチ内のデータごとにクラス確率に変換する
output_probs = output_logits.softmax(dim=1)

#バッチからインデックス 0 のデータを取り出して
#結果を表示する
class_id = output_probs[0].argmax().item()
score = output_probs[0][class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")