from tasks import *

TASK_DICT = {"cluster":ClusterPredictionTask,
 "descriptors":DescriptorPredictionTask,
 "fingeprints":FingerprintPredictionTask,
 "mask_fill":GraphMaskTask,
 "reconstruction":GraphReconstructionTask,
 "shuffling":ShufflingPredictionTask}