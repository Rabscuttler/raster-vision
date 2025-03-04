# Solar PV Array Detection Using Aerial Imagery of the UK
# =======================================================

# Using 4000x4000 aerial image files and polygon labels of existing
# arrays taken from Open Street Map

# Following the tiny_spacenet example and Rio Chip Classification
# https://github.com/azavea/raster-vision-examples/blob/master/examples/spacenet/rio/chip_classification.py 


# To run
# Check CLI help for useful args
# https://docs.rastervision.io/en/0.9/cli.html
# vars need -argument -a flag.
# e.g. -r Rerun commands, regardless if their output files already exist.


# rastervision run local -p pv_classification.py -a test True

# Check tensorboard
# tensorboard --logdir /opt/data/rv/classification/train



import re
import random
import os
from os.path import join

import rastervision as rv
from rastervision.utils.files import list_paths
from rastervision.augmentor import OversamplingAugmentor

##########################################
# Utils
##########################################
def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x

##########################################
# Experiment
##########################################
class SolarExperimentSet(rv.ExperimentSet):
    def exp_main(self, test=False):
        # docker filepath mounted to my data directory
        base_uri = '/opt/data/labels2'

        raster_uri = base_uri # rasters and labels in same directory for now
        label_uri = base_uri

        # Find all of the image ids that have associated images and labels. Collect
        # these values to use as our scene ids.
        # TODO use PV Array dataframe to select these
        label_paths = list_paths(label_uri, ext='.geojson')
        scene_ids = [x.split('.')[-2].split('/')[-1] for x in label_paths]

        scene2_ids = [
         'so9051_rgb_250_04', 'so9265_rgb_250_05', 'sp3590_rgb_250_04',
         'sj7304_rgb_250_04', 'su1385_rgb_250_06', 'st0709_rgb_250_05',
         'sj9004_rgb_250_05', 'st8022_rgb_250_05', 'st8303_rgb_250_05',
         'sj9402_rgb_250_05', 'so9078_rgb_250_06', 'sj9003_rgb_250_05',
         'sk0003_rgb_250_05', 'st8468_rgb_250_04', 'st6980_rgb_250_04',
         'su0883_rgb_250_05', 'su0983_rgb_250_05', 'so9249_rgb_250_05',
         'su1478_rgb_250_04', 'su1377_rgb_250_04', 'sj9002_rgb_250_06',
         'sj8903_rgb_250_04', 'sj9902_rgb_250_05', 'sj9602_rgb_250_05',
         'tg2827_rgb_250_04', 'sj9702_rgb_250_05', 'sj9803_rgb_250_04',
         'sj9802_rgb_250_05', 'sk0504_rgb_250_04', 'sk0302_rgb_250_05',
         'sk0306_rgb_250_04', 'sk0206_rgb_250_04', 'sk0207_rgb_250_04',
         'sk0503_rgb_250_04', 'sj9903_rgb_250_04', 'sk0202_rgb_250_06',
         'sk0309_rgb_250_03', 'sk0605_rgb_250_04', 'sk0405_rgb_250_04',
         'sk0404_rgb_250_04', 'sk0502_rgb_250_05', 'st5071_rgb_250_05',
         'sp3293_rgb_250_03', 'sy7691_rgb_250_05', 'sp3294_rgb_250_03',
         'sp3892_rgb_250_05', 'sp3690_rgb_250_04', 'st9979_rgb_250_05',
         'se6154_rgb_250_03', 'so8476_rgb_250_06', 'so8072_rgb_250_04',
         'so7972_rgb_250_04', 'sp3491_rgb_250_03', 'sp3490_rgb_250_03',
         'sp3291_rgb_250_03', 'sp3292_rgb_250_03', 'sp3492_rgb_250_03',
         'sk0212_rgb_250_03', 'so7878_rgb_250_06', 'tl1239_rgb_250_03',
         'su0972_rgb_250_03', 'st1532_rgb_250_04', 'so7556_rgb_250_05',
         'st7091_rgb_250_07', 'sn2040_rgb_250_04', 'so7371_rgb_250_04',
         'tl6064_rgb_250_05', 'so9255_rgb_250_05', 'st1826_rgb_250_04',
         'st1528_rgb_250_04', 'st1629_rgb_250_04', 'st0727_rgb_250_04',
         'st0827_rgb_250_04', 'st0928_rgb_250_04', 'st0930_rgb_250_04',
         'st0929_rgb_250_04', 'st0832_rgb_250_05', 'tl1750_rgb_250_03',
         'st2322_rgb_250_05', 'st1623_rgb_250_04', 'st1523_rgb_250_04',
         'st1624_rgb_250_04', 'st1424_rgb_250_04', 'st1421_rgb_250_05',
         'sp3793_rgb_250_04', 'sp3792_rgb_250_04', 'sj9912_rgb_250_03',
         'sk2347_rgb_250_05', 'sp3391_rgb_250_03', 'tl1846_rgb_250_03',
         'sp5177_rgb_250_03', 'sn3251_rgb_250_04', 'sp3693_rgb_250_04',
         'st2014_rgb_250_06', 'st2015_rgb_250_06', 'st2115_rgb_250_05',
         'st2114_rgb_250_05', 'sn4257_rgb_250_04', 'su4223_rgb_250_04',
         'su4323_rgb_250_04', 'tl3068_rgb_250_04', 'sp5178_rgb_250_03',
         'sp3791_rgb_250_04', 'st3689_rgb_250_03', 'st3789_rgb_250_03',
         'st0411_rgb_250_04', 'st0212_rgb_250_04', 'st0112_rgb_250_04',
         'st0211_rgb_250_04', 'st0111_rgb_250_04', 'st0209_rgb_250_05',
         'st0210_rgb_250_05', 'sj6714_rgb_250_04', 'sp3893_rgb_250_05',
         'su6712_rgb_250_04', 'su6713_rgb_250_04', 'st9363_rgb_250_04',
         'st9463_rgb_250_04', 'nr3059_rgb_250_03', 'st8576_rgb_250_03',
         'sp7948_rgb_250_04', 'sp6138_rgb_250_07', 'tl2276_rgb_250_04',
         'sm9817_rgb_250_04', 'sm9816_rgb_250_04', 'sm9716_rgb_250_04',
         'sm9616_rgb_250_04', 'sm9818_rgb_250_04', 'sm9009_rgb_250_04',
         'sm9721_rgb_250_05', 'sm9720_rgb_250_05', 'sm9101_rgb_250_04',
         'sm9201_rgb_250_04', 'sm9010_rgb_250_04', 'sm9109_rgb_250_04',
         'sn6502_rgb_250_04', 'sn6601_rgb_250_04', 'sn6201_rgb_250_04',
         'sn6202_rgb_250_04', 'st6788_rgb_250_05', 'st6688_rgb_250_05',
         'st6689_rgb_250_06', 'su0807_rgb_250_05', 'su0806_rgb_250_05',
         'sz0998_rgb_250_05', 'sz1099_rgb_250_05', 'su3743_rgb_250_04',
         'su3744_rgb_250_04', 'su6509_rgb_250_04', 'su6409_rgb_250_04',
         'su6410_rgb_250_04', 'su5413_rgb_250_04', 'su2088_rgb_250_04',
         'su5703_rgb_250_04', 'su5603_rgb_250_04', 'su5604_rgb_250_04',
         'st7642_rgb_250_06', 'st7744_rgb_250_05', 'st6728_rgb_250_05',
         'st8558_rgb_250_04', 'st2735_rgb_250_04', 'tl4990_rgb_250_05',
         'sm7209_rgb_250_04', 'st8864_rgb_250_04', 'tg5013_rgb_250_04',
         'st1198_rgb_250_04', 'st1298_rgb_250_04', 'st1722_rgb_250_04',
         'tq1078_rgb_250_05', 'su6401_rgb_250_04', 'st8753_rgb_250_04',
         'st8455_rgb_250_05', 'st8660_rgb_250_04', 'st8760_rgb_250_04',
         'st8765_rgb_250_04', 'sp7638_rgb_250_05', 'tl6332_rgb_250_04',
         'st8705_rgb_250_05', 'sy3297_rgb_250_06', 'sy3498_rgb_250_06',
         'se3636_rgb_250_01', 'st6578_rgb_250_05', 'st6478_rgb_250_05',
         'st5479_rgb_250_06', 'se2931_rgb_250_02', 'sd6835_rgb_250_01',
         'st2228_rgb_250_05', 'st2227_rgb_250_05']

        # Experiment label and root directory for output 
        exp_id = 'pv-classification'
        root_uri = '/opt/data/rv/test3'

        # num_steps = 1e4 # 1e5 takes too long
        num_epochs = 20
        batch_size = 16 
        debug = True
       
        test = str_to_bool(test)
        if test:
            print("***************** TEST MODE *****************")
            exp_id += '-test'
            # num_steps = 100
            num_epochs = 1
            batch_size = 1
            debug = True
            train_ids = scene_ids
            val_ids = scene_ids
            scene_ids = scene_ids[0:5]
       
       # Split the data into training and validation sets:
        # Randomize the order of all scene ids
        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)

        # Set scenes
        num_train_ids = round(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]
        # train_ids = scene_ids
        # val_ids = scene_ids

        # ------------- TASK -------------

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
            .with_chip_size(200) \
            .with_classes({
                'pv': (1, 'yellow'),
                'background': (2, 'black')
            })\
            .build()


        # # ------------- BACKEND -------------
        # Configuration options for different models and tasks:
        # https://github.com/azavea/raster-vision/blob/60f741e30a016f25d2643a9b32916adb22e42d50/rastervision/backend/model_defaults.json

        backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
            .with_task(task) \
            .with_debug(debug) \
            .with_batch_size(batch_size) \
            .with_num_epochs(num_epochs) \
            .with_model_defaults(rv.RESNET50_IMAGENET) \
            .with_config({
                'trainer': {
                    'options': {
                        'saveBest': True,
                        'lrSchedule': [
                            {
                              'epoch': 0,
                              'lr': 0.0005
                            },
                            {
                              'epoch': 10,
                              'lr': 0.0001
                            },
                            {
                              'epoch': 15,
                              'lr': 0.00001
                            }
                        ]
                    }
                 }
            }, set_missing_keys=True) \
            .build()

        # ------------- Make Scenes -------------
        # We will use this function to create a list of scenes that we will pass
        # to the DataSetConfig builder.
        def make_scene(id):
            """Make a SceneConfig object for each image/label pair
            Args:
                id (str): The id that corresponds to both the .jpg image source
                    and .geojson label source for a given scene
            Returns:
                rv.data.SceneConfig: a SceneConfig object which is composed of
                    images, labels and optionally AOIs
            """
            # Find the uri for the image associated with this is
            image_uri = os.path.join(raster_uri,
                                           '{}.jpg'.format(id))
            # Construct a raster source from an image uri that can be handled by Rasterio.
            # We also specify the order of image channels by their indices and add a
            # stats transformer which normalizes pixel values into uint8.
            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            label = os.path.join(label_uri, '{}.geojson'.format(id))
            
            # Build our classification labels.
            # IOA Threshold means minimum percentage of IOA of polygon and cell
            # infer_cells means label source infers polygons and labels
            # from the vector source.
            # with_pick_min_class_id True means will always pick PV if it is present
            # as it is the smaller id number
            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                .with_uri(label) \
                .with_ioa_thresh(0.01) \
                .with_use_intersection_over_cell(False) \
                .with_pick_min_class_id(True) \
                .with_background_class_id(2) \
                .with_infer_cells(True) \
                .build()

            # Finally we can build a scene config object using the scene id and the
            # configs we just defined
            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene

        # Create lists of train and test scene configs
        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        # ------------- DATASET -------------
        # Construct a DataSet config using the lists of train and
        # validation scenes
        augmentor = 
        
        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .with_augmentor(augmentor)
            .build()

        # ------------- ANALYZE -------------
        # We will need to convert this imagery from uint16 to uint8
        # in order to use it. We specified that this conversion should take place
        # when we built the train raster source but that process will require
        # dataset-level statistics. To get these stats we need to create an
        # analyzer.
        # Use small sample prob so this step doesn't take ages.
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
            .with_sample_prob(0.05) \
            .build()

        # ------------- EXPERIMENT -------------
        experiment = rv.ExperimentConfig.builder() \
            .with_id(exp_id) \
            .with_task(task) \
            .with_backend(backend) \
            .with_analyzer(analyzer) \
            .with_dataset(dataset) \
            .with_root_uri(root_uri) \
            .build()

        return experiment


if __name__ == '__name__':
    rv.main()
