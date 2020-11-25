# annotation-automation

Dataloop FaaS example for functions that performs automations on items and annotations.

## Download package artifacts

You need to download the artifacts of this package from:
https://storage.googleapis.com/dtlpy/model_assets/automation-package/artifacts.zip

## SDK Installation

You need to have dtlpy installed, if don't already, install it by running:


```bash
pip install dtlpy --upgrade
```

## Usage

### CLI

```bash
cd <this directory>

dlp projects checkout --project-name <name of the project>

dlp packages push --checkout

dlp packages deploy --checkout
```
### SDK

```python
import dtlpy as dl

package_name = 'annotation-automation'
project_name = 'My project'
project = dl.projects.get(project_name=project_name)

##########################
# define package modules #
##########################
modules = [
    dl.PackageModule(
        init_inputs=[
            dl.FunctionIO(type='Json', name='project_name')
        ],
        name='default',
        entry_point='main.py',
        functions=[
            dl.PackageFunction(
                inputs=[
                    dl.FunctionIO(type="Item", name="item"),
                    dl.FunctionIO(type="Json", name="annotations"),
                    dl.FunctionIO(type="Json", name="config")
                ],
                name='bbox_to_segmentation',
                description='Converts a bounding box into a segmentation'),
            dl.PackageFunction(
                inputs=[
                    dl.FunctionIO(type="Item", name="item"),
                    dl.FunctionIO(type="Json", name="annotations"),
                    dl.FunctionIO(type="Json", name="config")
                ],
                name='four_points_to_segmentation',
                description='Converts a four points closed polygon into a segmentation'),
            dl.PackageFunction(
                inputs=[
                    dl.FunctionIO(type="Item", name="item"),
                    dl.FunctionIO(type="Json", name="config")
                ],
                name='item_auto_annotation_automation',
                description='Auto annotate a platform item using maskrcnn')
        ]
    )
]

################
# push package #
################
package = project.packages.push(package_name=package_name, modules=modules)

####################
# upload artifacts #
####################
project.artifacts.upload(filepath='/path_to_weights/dextr_pascal-sbd.h5',
                         package_name=package_name)
project.artifacts.upload(filepath='/path_to_weights/mask_rcnn_coco.h5',
                         package_name=package_name)
project.artifacts.upload(filepath='/path_to_points/SiamMask_DAVIS.pth',
                         package_name=package_name)
project.artifacts.upload(filepath='/path_to_points/cityscapes_checkpoint.pt',
                         package_name=package_name)

##################
# deploy service #
##################
service = package.services.deploy(service_name=package.name,
                                  runtime={
                                      'numReplicas': 1,
                                      'concurrency': 5,
                                      'podType': dl.InstanceCatalog.GPU_K80_S,
                                      # use our docker image or build one of your own
                                      # see Dockerfile for more info
                                      'runnerImage':
                                          'gcr.io/viewo-g/piper/agent/runner/gpu/box2seg-dextr-maskrcnn:latest'
                                  },
                                  module_name='default',
                                  init_input={'project_name': project.name}
                                  )
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
