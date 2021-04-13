PX=160

## make img_download PX=[default: 160 px]
##   Downloads fastai imagenette dataset of specified pixels
img_download:
	mkdir -p images
	curl -o images/original.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-$(PX).tgz
	tar -xvf images/original.tgz -C images
	mv images/imagenette2-$(PX) images/original