-----------------------------------------------------------------------------------
## Our_GAN

	# Env install 

	# Directory Structure

		Our_GAN
		|
		|--dataloaders
		|	|
		|	|--CocoStuffDataset.py : for Mapillary 
		|	
		|		Hyperparameter	
		|		--load_size 	: 256
		|		--crop_size 	: 256
		|		--label_nc		: 67 + 1(edge)
		|		--semantic_nc	: 68 + 1(fake)
		|		--aspect_ratio	: 1.0 (Res:256*256)
		|	
		|	|--CityscapesDataset.py  : for Cityscapes 
		|	
		|		Hyperparameter	
		|		--load_size 	: 512
		|		--crop_size 	: 512
		|		--label_nc		: 34 + 1(edge)
		|		--semantic_nc	: 35 + 1(fake)
		|		--aspect_ratio	: 2.0 (Res:512*256)
		|
		|--models
		|	|
		|	|--diffaugment.py : Differentiable Augmentation Function
		|		|
		|		|	* Use brightness, saturation, contrast, translation
		|	|
		|	|--discriminator.py : Discriminator architecture (U-Net)
		|	|
		|	|--generator.py : Generator architecture 
		|		|
		|		|	* SPADE > CLADE, add Dual attention modules
		|	|
		|	|--losses.py : Define loss function
		|	|
		|	|--models.py : GAN architecture
		|		|
		|		|	* Include update G and D & preprocess images
		|	|
		|	|--norms.py : Normalization
		|		|
		|		|	* CLADE architecture
		|
		|--utils : plot loss curve & save model & FID evaluation
		|
		|--test.py : test script
		|
		|--train.py : training script
		|		|
		|		|	* Probabilty decay														
		|
		|--config.py
		|
		|	Hyperparameter
		|	
		|	* General options
		|	--name 			: Name of the experiment
		|	--gpu_ids 		: gpu ids
		|	--batch_size	: input batch size
		|	--dataroot 		: path to dataset root
		|	--dataset_mode	: this option indicates which dataset should be loaded
		|	--batch_size	: input batch size
		|	
		|	* for generator
		|	--no_3dnoise 	: if specified, do *not* concatenate noise to label maps
		|	--z_dim 		: dimension of the latent z vector
		|	
		|	* for train
		|	--freq_print 	: frequency of showing training results
		|	--freq_save_ckpt: frequency of saving the checkpoints
		|	--freq_save_latest: frequency of saving the latest model
		|	--freq_smooth_loss: smoothing window for loss visualization
		|	--freq_save_loss: frequency of loss plot updates
		|	--freq_fid		: frequency of saving the fid score (in training iterations)
		|	--continue_train: resume previously interrupted training
		|	--which_iter	: which epoch to load when continue_train
		|	--num_epochs	: number of epochs to train
		|	--beta1			: momentum term of adam
		|	--beta2			: momentum term of adam
		|	--lr_g			: G learning rate, default=0.0001
		|	--lr_d			: D learning rate, default=0.0004
		|	--lambda_labelmix: weight for LabelMix regularization
		|	--no_labelmix	: if specified, do *not* use LabelMix
		|	--no_balancing_inloss: if specified, do *not* use class balancing in the loss function
		|	--lambda_DiffAugPA: probabilistic attenuation decay rate
		|	
		|	* for test
		|	--results_dir	: saves testing results here
		|	--ckpt_iter		: which epoch to load to evaluate a model

	# How to use

		1. conda activate SPADE

		2. Training

			# To train cityscapes dataset

			python train.py --dataset_mode cityscapes --dataroot 'XXX/cityscapes' --name XXX --gpu_ids 0

			# To train Mapillary dataset

			python train.py --dataset_mode coco --dataroot 'XXX/mapillary_coco' --name XXX --gpu_ids 0

			* example : python train.py --dataset_mode cityscapes --dataroot '/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/Datasets/cityscapes' --name test --gpu_ids 0

			notes : models and logs will be './checkpoints'

		3. Test

			# To test cityscapes dataset

			python test.py --dataset_mode cityscapes --dataroot 'XXX/cityscapes' --name XXX --gpu_ids 0

			# To test Mapillary dataset

			python test.py --dataset_mode coco --dataroot 'XXX/mapillary_coco' --name XXX --gpu_ids 0

			* example : python test.py --dataset_mode cityscapes --dataroot '/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/Datasets/cityscapes' --name test --gpu_ids 0

			notes : result will be './results'

		4. Trained model name : aaa(cityscapes) 

-----------------------------------------------------------------------------------

## PointRend_detectron2

	# Env install

		!pip install pyyaml==5.1
		!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
		!pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
		!git clone --branch v0.4 https://github.com/facebookresearch/detectron2.git detectron2_repo

	# Directory Structure

		PointRend_detectron2
		| 
		|--detectron2_repo : (clone from github)
		| 
		|--test.py (Generate instance maps from source image folder)
		|
		|	Hyperparameter
		|	--dataroot : The root of source dataset images
		|	--saveroot : The save root of generated instance map
		|	--res	   : Width,Heigth
		|
		|--test.sh : The shell script

	# How to use

		1. conda activate detectorn2 (!)
		2. python test.py --dataroot 'XXX' --saveroot 'XXX' --res 1080,1920

-----------------------------------------------------------------------------------

## semantic-segmentation

	# Env install

		pyyaml>=5.1.1
		coolname>=1.1.0
		tabulate>=0.8.3
		tensorboardX>=1.4
		runx==0.0.6

		!git clone https://github.com/NVIDIA/semantic-segmentation.git

	# Directory Structure

		semantic-segmentation : (clone from github)
		|
		|--scripts
		|	|
		|	|--dump_folder.yml (The function to generate semantic segmentations)
		|		|
		|		|--dataset : Mapillary dataset (65 labels)
		|		|
		|		|--eval_folder : The root of inference image folder
		|	
		|--logs
		|	|
		|	|--dump_folder
		|		|
		|		|--new date folder
		|			|
		|			|--logs
		|				|
		|				|--dump_folder
		|					|
		|					|--new date folder
		|						|
		|						|--best_images
		|							|
		|							|--image name_predcition.png : (The generated semantic segmentation)


	# How to use

		1. conda activate SPADE (!)
		2. put inference images to '/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/semantic-segmentation/imgs/test_imgs'
		3. python -m runx.runx scripts/dump_folder.yml -i
		4. generated images will be '/mnt/nvideo1/M10802131_Zhenyu_Li_Graduation/Proposed_work/semantic-segmentation/logs/dump_folder/XXX/logs/dump_folder/XXX/best_images/'
		5. copy all XXX_prediction.png to new folder
		6. rename 