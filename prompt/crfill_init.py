import argparse, sys



def initialize(parser):
    # experiment specifics
    # parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

    # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    # for setting inputs
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
    parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

    # for displays
    parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

    # for generator
    parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for instance-wise features
    # I added
    parser.add_argument('--norm_type', type=int, default=1, help='1:my bug, 2:org')
    parser.add_argument('--use_th', action='store_true', help='th or th boundary')
    parser.add_argument('--th', type=float, default=0.5, help='1:my bug, 2:org')

    return parser