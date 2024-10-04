from pycocotools.coco import COCO

dataDir = 'path_to_coco_data'
dataType = 'train2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'

coco = COCO(annFile)
