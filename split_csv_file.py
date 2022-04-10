import pandas as pd
val_num = 2
sample_submission = pd.read_csv('dataset/label.csv')
train_idx,val_idx = 0,0
train = pd.DataFrame(columns=['filename','category'])
val = pd.DataFrame(columns=['filename','category'])
past,cnt = -1,0
for idx, filename in enumerate(sample_submission['filename']):
    label = sample_submission['category'][idx]
    if label == past:
        cnt += 1
    else:
        cnt = 0
    if cnt < val_num:
        val = val.append({'filename': filename,'category': label}, ignore_index=True)
        print(label)
    else:
        train = train.append({'filename': filename,'category': label}, ignore_index=True)
    past = label

train.to_csv(("./dataset/train_label.csv"), index=False)
val.to_csv(("./dataset/val_label.csv"), index=False)
