#train
# python src/classifier.py TRAIN pretrained/train/ pretrained/20180402-114759/20180402-114759.pb pretrained/my_classifier.pkl --batch_size 1000

#test
# python src/classifier.py CLASSIFY pretrained/train/ pretrained/20180402-114759/20180402-114759.pb pretrained/my_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

#predict
# python tmp/predict.py pretrained/sana/SANA_6.jpg pretrained/20180402-114759/20180402-114759.pb pretrained/my_classifier.pkl
