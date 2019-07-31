# Solar PV Array Detection Using Aerial Imagery of the UK
# =======================================================

# Using 4000x4000 aerial image files and polygon labels of existing
# arrays taken from Open Street Map

# Following the tiny_spacenet example and Vegas simple_segmentation
# https://github.com/azavea/raster-vision-examples/blob/0.9/examples/spacenet/vegas/simple_segmentation.py


# To run:

# rastervision run local -p pv_detection_02.py -a test True




import re
import random
import os
from os.path import join

import rastervision as rv
from rastervision.utils.files import list_paths

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
        base_uri = '/opt/data/labels'

        raster_uri = base_uri # rasters and labels in same directory for now
        label_uri = base_uri

        # Find all of the image ids that have associated images and labels. Collect
        # these values to use as our scene ids.
        # TODO use PV Array dataframe to select these
        label_paths = list_paths(label_uri, ext='.geojson')
        scene_ids = [x.split('.')[-2].split('/')[-1] for x in label_paths]

        # Experiment label, used to label config files
        exp_id = 'pv-detection-1'

        # Number of times passing a batch of images through the model
        num_steps = 1e5
        # Number of images in each batch
        batch_size = 8
        # Specify whether or not to make debug chips (a zipped sample of png chips
        # that you can examine to help debug the chipping process)
        debug = True

        # This experiment includes an option to run a small test experiment before
        # running the whole thing. You can set this using the 'test' parameter. If
        # this parameter is set to True it will run a tiny test example with a new
        # experiment id. This will be small enough to run locally. It is recommended
        # to run a test example locally before submitting the whole experiment to AWs
        # Batch.
        test = str_to_bool(test)
        if test:
            print("***************** TEST MODE *****************")
            exp_id += '-test'
            num_steps = 1
            batch_size = 1
            debug = True
            scene_ids = scene_ids[0:10]

        # Split the data into training and validation sets:
        # Randomize the order of all scene ids
        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)

        # Figure out how many scenes make up 80% of the whole set
        num_train_ids = round(len(scene_ids) * 0.8)
        # Split the scene ids into training and validation lists
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]


        # ------------- TASK -------------

        # task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
        #     .with_chip_size(300) \
        #     .with_classes({
        #     'pv': (1, 'yellow'),
        #     # 'background': (2, 'black')
        #     }) \
        #     .with_chip_options(
        #                         chips_per_scene=50,
        #                         debug_chip_probability=0.1,
        #                         negative_survival_probability=1.0,
        #                         target_classes=[1],
        #                         target_count_threshold=1000) \
        #     .build()

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
            .with_chip_size(300) \
            .with_classes({
            'pv': (1, 'yellow'),
            # 'background': (2, 'black')
            }) \
            .build()

        # # ------------- BACKEND -------------
        # Configuration options for different models and tasks:
        # https://github.com/azavea/raster-vision/blob/60f741e30a016f25d2643a9b32916adb22e42d50/rastervision/backend/model_defaults.json

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
            .with_task(task) \
            .with_debug(debug) \
            .with_batch_size(num_steps) \
            .with_num_steps(batch_size) \
            .with_model_defaults(rv.MOBILENET_V2) \
            .with_train_options(replace_model=False) \
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
            train_image_uri = os.path.join(raster_uri,
                                           '{}.jpg'.format(id))
            # Construct a raster source from an image uri that can be handled by Rasterio.
            # We also specify the order of image channels by their indices and add a
            # stats transformer which normalizes pixel values into uint8.
            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            # Next create a label source config to pair with the raster source:
            # define the geojson label source uri
            vector_source = os.path.join(
                label_uri, '{}.geojson'.format(id))

            # Since this is a semantic segmentation experiment and the labels
            # are distributed in a vector-based GeoJSON format, we need to rasterize
            # the labels. We create  aRasterSourceConfigBuilder using
            # `rv.RASTERIZED_SOURCE`
            # indicating that it will come from a vector source. We then specify the uri
            # of the vector source and (in the 'with_rasterizer_options' method) the id
            # of the pixel class we would like to use as background.
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(vector_source) \
                .with_rasterizer_options(2) \
                .build()


            # Create a semantic segmentation label source from rasterized source config
            # that we built in the previous line.
            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
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
        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        # ------------- ANALYZE -------------
        # We will need to convert this imagery from uint16 to uint8
        # in order to use it. We specified that this conversion should take place
        # when we built the train raster source but that process will require
        # dataset-level statistics. To get these stats we need to create an
        # analyzer.
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # ------------- EXPERIMENT -------------
        experiment = rv.ExperimentConfig.builder() \
            .with_id(exp_id) \
            .with_task(task) \
            .with_backend(backend) \
            .with_analyzer(analyzer) \
            .with_dataset(dataset) \
            .with_root_uri('/opt/data/rv') \
            .build()

        return experiment


if __name__ == '__name__':
    rv.main()
