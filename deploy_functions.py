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
