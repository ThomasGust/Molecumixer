from tasks.common import Task

from tasks.clusters import Clusterer, ClusterPredictionModel
from tasks.descriptors import DescriptorCalculator, DescriptorPredictionModel
from tasks.fingerprints import FingerprintCalculator, FingerprintPredictionModel
from tasks.mask_fill import GraphMasker, GraphMaskPredictionModel
from tasks.reconstruction import GVAE, GraphDiscriminator
from tasks.shuffling import NodeShuffler, ShufflingModel