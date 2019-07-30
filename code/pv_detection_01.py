# Solar PV Array Detection Using Aerial Imagery of the UK
# =======================================================

# Using 4000x4000 aerial image files and polygon labels of existing
# arrays taken from Open Street Map

# Following the tiny_spacenet example

import rastervision as rv

class SolarExperimentSet(rv.ExperimentSet):
    def exp_main(self):
        base_uri = ('/opt/data') # docker filepath mounted to my data directory
        train_image_uri = '{base_uri}/tq2866_l2_250_01.jpg'.format(base_uri)
        train_label_uri = '{base_uri}/tq2866_l2_250_01.jpg'.format(base_uri)
        val_image_uri = '{base_uri}/'.format(base_uri)
        val_label_uri = '{base_uri}/'.format(base_uri)

        # ------------- TASK -------------

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(250) \
                            .with_chip_options(chips_per_scene=256) \
                            .with_classes({
                                'pv' : (1, 'yellow')
                            }) \
                            .build()

        # # ------------- BACKEND -------------
        #
        # backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
        #   .with_task(task) \
        #   .with_debug(True) \
        #   .with_batch_size(1) \
        #   .with_num_steps(1) \
        #   .with_model_defaults(rv.MOBILENET_V2) \
        #   .build()

        return


if __name__ == '__name__':
    rv.main()
