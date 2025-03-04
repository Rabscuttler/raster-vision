# Solar PV Array Detection Using Aerial Imagery of the UK
# =======================================================

# Using 4000x4000 aerial image files and polygon labels of existing
# arrays taken from Open Street Map

# Following the tiny_spacenet example

import rastervision as rv

class SolarExperimentSet(rv.ExperimentSet):
    def exp_main(self):
        base_uri = ('opt/data/labels') # docker filepath mounted to my data directory
        train_image_uri = '/{}/sp1182_l2_250_04.jpg'.format(base_uri)
        train_label_uri = '/{}/sp1182_l2_250_04.geojson'.format(base_uri)
        val_image_uri = '/{}/sp3759_l2_250_03.jpg'.format(base_uri)
        val_label_uri = '/{}/sp3759_l2_250_03.geojson'.format(base_uri)
        channel_order = [0, 1, 2] # Keeping this but don't understand if needed
        background_class_id = 2 # presumably important?

        # ------------- TASK -------------

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_chip_options(chips_per_scene=50) \
                            .with_classes({
                                'pv': (1, 'yellow'),
                                'background': (2, 'black')
                            }) \
                            .build()

        # # ------------- BACKEND -------------
        
        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_debug(True) \
                                  .with_batch_size(1) \
                                  .with_num_steps(1) \
                                  .with_model_defaults(rv.MOBILENET_V2) \
                                  .build()

        # ------------- TRAINING -------------

        train_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                   .with_uri(train_image_uri) \
                                                   .with_channel_order(channel_order) \
                                                   .with_stats_transformer() \
                                                   .build()

        train_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                         .with_vector_source(train_label_uri) \
                                                         .with_rasterizer_options(background_class_id) \
                                                         .build()
        train_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                                 .with_raster_source(train_label_raster_source) \
                                                 .build()

        train_scene =  rv.SceneConfig.builder() \
                                     .with_task(task) \
                                     .with_id('train_scene') \
                                     .with_raster_source(train_raster_source) \
                                     .with_label_source(train_label_source) \
                                     .build()

        # ------------- VALIDATION -------------

        val_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                 .with_uri(val_image_uri) \
                                                 .with_channel_order(channel_order) \
                                                 .with_stats_transformer() \
                                                 .build()

        val_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                       .with_vector_source(val_label_uri) \
                                                       .with_rasterizer_options(background_class_id) \
                                                       .build()
        val_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                               .with_raster_source(val_label_raster_source) \
                                               .build()

        val_scene = rv.SceneConfig.builder() \
                                  .with_task(task) \
                                  .with_id('val_scene') \
                                  .with_raster_source(val_raster_source) \
                                  .with_label_source(val_label_source) \
                                  .build()
        

        # ------------- DATASET -------------

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(train_scene) \
                                  .with_validation_scene(val_scene) \
                                  .build()

        # ------------- EXPERIMENT -------------

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('pv-detection-experiment') \
                                        .with_root_uri('/opt/data/rv') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_stats_analyzer() \
                                        .build()

        return experiment


if __name__ == '__name__':
    rv.main()
