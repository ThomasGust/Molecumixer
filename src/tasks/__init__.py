from tasks.common import Task

from tasks.clusters import Clusterer, ClusterPredictionModel, ClusterPredictionTask
from tasks.descriptors import DescriptorCalculator, DescriptorPredictionModel, DescriptorPredictionTask
from tasks.fingerprints import FingerprintCalculator, FingerprintPredictionModel, FingerprintPredictionTask
from tasks.mask_fill import GraphMasker, GraphMaskPredictionModel, GraphMaskTask
from tasks.reconstruction import GVAE, GraphDiscriminator, GraphReconstructionTask
from tasks.shuffling import NodeShuffler, ShufflingModel, ShufflingPredictionTask