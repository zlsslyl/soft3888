set -e

isc ping

isc train cifar100-resnet50.isc
isc train fashion_mnist.isc
isc train resnet50.isc
isc train resnet152.isc
isc train efficientnet_b0.isc
isc train efficientnet_b7.isc
isc train efficientnetv2_s.isc
isc train efficientnetv2_xl.isc
isc train vit_base_patch16_224.isc
isc train vit_large_patch16_224.isc
isc train fcn_resnet101.isc
isc train deeplabv3_mobilenet_v3_large.isc