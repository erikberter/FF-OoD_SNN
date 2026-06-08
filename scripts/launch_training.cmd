python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "784" --batch_size "512" --threshold "6" --negative_threshold "2" --alpha "1" --beta "1" --epochs "10" --lr "0.001"
python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "784" --batch_size "512" --threshold "6" --negative_threshold "2" --alpha "1" --beta "1" --epochs "10" --lr "0.002" --use_snn
python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "784" --batch_size "512" --threshold "0.3" --negative_threshold "0.1" --alpha "5" --beta "5" --epochs "10" --lr "0.002" --use_snn --bounded_goodness


python train.py --dataset "CIFAR10" --neurons_per_layer "1024" --input_size "2048" --batch_size "512" --threshold "12"  --negative_threshold "12" --alpha "1" --beta "1" --feature_extractor "SmallResNetCIFAR" --epochs "20"
python train.py --dataset "CIFAR10" --neurons_per_layer "1024" --input_size "2048" --batch_size "512" --threshold "12"  --negative_threshold "12" --alpha "1" --beta "1" --feature_extractor "SmallResNetCIFAR" --epochs "20" --use_snn


python train.py --dataset "SVHN" --neurons_per_layer "1024" --input_size "2048" --batch_size "512" --threshold "12"  --negative_threshold "12" --alpha "1" --beta "1" --feature_extractor "SmallResNetSVHN" --epochs "20"
python train.py --dataset "SVHN" --neurons_per_layer "1024" --input_size "2048" --batch_size "512" --threshold "12"  --negative_threshold "12" --alpha "1" --beta "1" --feature_extractor "SmallResNetSVHN" --epochs "20" --use_snn
python train.py --dataset "SVHN" --neurons_per_layer "1024" --input_size "2048" --batch_size "512" --threshold "0.3"  --negative_threshold "0.1" --alpha "5" --beta "5"  --feature_extractor "SmallResNetSVHN" --epochs "20" --use_snn --bounded_goodness