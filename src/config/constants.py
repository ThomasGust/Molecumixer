import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

X_MAP = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

E_MAP = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

MAX_MOLECULE_SIZE = 250
#MAX_EDGES = int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1)/2)) The quadratic approach is naive.
# In reality, atoms can only form up to 7 bonds. Also, because the bonds are bidirectional, we can reduce that
# to 7/2 * MAX_MOLECULE_SIZE
MAX_EDGES = int(3.5*MAX_MOLECULE_SIZE)+1
DISABLE_RDKIT_WARNINGS = False

ELEMENT_BASE = {
    # number: name symbol common_ions uncommon_ions
    # ion info comes from Wikipedia: list of oxidation states of the elements.
    0: ['Neutron',     'n',  [],         []],
    1: ['Hydrogen',    'H',  [-1, 1],    []],
    2: ['Helium',      'He', [],         [1, 2]],  # +1,+2  http://periodic.lanl.gov/2.shtml
    3: ['Lithium',     'Li', [1],        []],
    4: ['Beryllium',   'Be', [2],        [1]],
    5: ['Boron',       'B',  [3],        [-5, -1, 1, 2]],
    6: ['Carbon',      'C',  [-4, -3, -2, -1, 1, 2, 3, 4], []],
    7: ['Nitrogen',    'N',  [-3, 3, 5], [-2, -1, 1, 2, 4]],
    8: ['Oxygen',      'O',  [-2],       [-1, 1, 2]],
    9: ['Fluorine',    'F',  [-1],       []],
    10: ['Neon',       'Ne', [],         []],
    11: ['Sodium',     'Na', [1],        [-1]],
    12: ['Magnesium',  'Mg', [2],        [1]],
    13: ['Aluminum',   'Al', [3],        [-2, -1, 1, 2]],
    14: ['Silicon',    'Si', [-4, 4],    [-3, -2, -1, 1, 2, 3]],
    15: ['Phosphorus', 'P',  [-3, 3, 5], [-2, -1, 1, 2, 4]],
    16: ['Sulfur',     'S',  [-2, 2, 4, 6],    [-1, 1, 3, 5]],
    17: ['Chlorine',   'Cl', [-1, 1, 3, 5, 7], [2, 4, 6]],
    18: ['Argon',      'Ar', [],         []],
    19: ['Potassium',  'K',  [1],        [-1]],
    20: ['Calcium',    'Ca', [2],        [1]],
    21: ['Scandium',   'Sc', [3],        [1, 2]],
    22: ['Titanium',   'Ti', [4],        [-2, -1, 1, 2, 3]],
    23: ['Vanadium',   'V',  [5],        [-3, -1, 1, 2, 3, 4]],
    24: ['Chromium',   'Cr', [3, 6],     [-4, -2, -1, 1, 2, 4, 5]],
    25: ['Manganese',  'Mn', [2, 4, 7],  [-3, -2, -1, 1, 3, 5, 6]],
    26: ['Iron',       'Fe', [2, 3, 6],  [-4, -2, -1, 1, 4, 5, 7]],
    27: ['Cobalt',     'Co', [2, 3],     [-3, -1, 1, 4, 5]],
    28: ['Nickel',     'Ni', [2],        [-2, -1, 1, 3, 4]],
    29: ['Copper',     'Cu', [2],        [-2, 1, 3, 4]],
    30: ['Zinc',       'Zn', [2],        [-2, 1]],
    31: ['Gallium',    'Ga', [3],        [-5, -4, -2, -1, 1, 2]],
    32: ['Germanium',  'Ge', [-4, 2, 4], [-3, -2, -1, 1, 3]],
    33: ['Arsenic',    'As', [-3, 3, 5], [-2, -1, 1, 2, 4]],
    34: ['Selenium',   'Se', [-2, 2, 4, 6], [-1, 1, 3, 5]],
    35: ['Bromine',    'Br', [-1, 1, 3, 5], [4, 7]],
    36: ['Krypton',    'Kr', [2],        []],
    37: ['Rubidium',   'Rb', [1],        [-1]],
    38: ['Strontium',  'Sr', [2],        [1]],
    39: ['Yttrium',    'Y',  [3],        [1, 2]],
    40: ['Zirconium',  'Zr', [4],        [-2, 1, 2, 3]],
    41: ['Niobium',    'Nb', [5],        [-3, -1, 1, 2, 3, 4]],
    42: ['Molybdenum', 'Mo', [4, 6],     [-4, -2, -1, 1, 2, 3, 5]],
    43: ['Technetium', 'Tc', [4, 7],     [-3, -1, 1, 2, 3, 5, 6]],
    44: ['Ruthenium',  'Ru', [3, 4],     [-4, -2, 1, 2, 5, 6, 7, 8]],
    45: ['Rhodium',    'Rh', [3],        [-3, -1, 1, 2, 4, 5, 6]],
    46: ['Palladium',  'Pd', [2, 4],     [1, 3, 5, 6]],
    47: ['Silver',     'Ag', [1],        [-2, -1, 2, 3, 4]],
    48: ['Cadmium',    'Cd', [2],        [-2, 1]],
    49: ['Indium',     'In', [3],        [-5, -2, -1, 1, 2]],
    50: ['Tin',        'Sn', [-4, 2, 4], [-3, -2, -1, 1, 3]],
    51: ['Antimony',   'Sb', [-3, 3, 5], [-2, -1, 1, 2, 4]],
    52: ['Tellurium',  'Te', [-2, 2, 4, 6], [-1, 1, 3, 5]],
    53: ['Iodine',     'I',  [-1, 1, 3, 5, 7], [4, 6]],
    54: ['Xenon',      'Xe', [2, 4, 6],  [8]],
    55: ['Cesium',     'Cs', [1],        [-1]],
    56: ['Barium',     'Ba', [2],        [1]],
    57: ['Lanthanum',  'La', [3],        [1, 2]],
    58: ['Cerium',     'Ce', [3, 4],     [2]],
    59: ['Praseodymium', 'Pr', [3],      [2, 4, 5]],
    60: ['Neodymium',  'Nd', [3],        [2, 4]],
    61: ['Promethium', 'Pm', [3],        [2]],
    62: ['Samarium',   'Sm', [3],        [2]],
    63: ['Europium',   'Eu', [2, 3],     []],
    64: ['Gadolinium', 'Gd', [3],        [1, 2]],
    65: ['Terbium',    'Tb', [3],        [1, 2, 4]],
    66: ['Dysprosium', 'Dy', [3],        [2, 4]],
    67: ['Holmium',    'Ho', [3],        [2]],
    68: ['Erbium',     'Er', [3],        [2]],
    69: ['Thulium',    'Tm', [3],        [2]],
    70: ['Ytterbium',  'Yb', [3],        [2]],
    71: ['Lutetium',   'Lu', [3],        [2]],
    72: ['Hafnium',    'Hf', [4],        [-2, 1, 2, 3]],
    73: ['Tantalum',   'Ta', [5],        [-3, -1, 1, 2, 3, 4]],
    74: ['Tungsten',   'W',  [4, 6],     [-4, -2, -1, 1, 2, 3, 5]],
    75: ['Rhenium',    'Re', [4],        [-3, -1, 1, 2, 3, 5, 6, 7]],
    76: ['Osmium',     'Os', [4],        [-4, -2, -1, 1, 2, 3, 5, 6, 7, 8]],
    77: ['Iridium',    'Ir', [3, 4],     [-3, -1, 1, 2, 5, 6, 7, 8, 9]],
    78: ['Platinum',   'Pt', [2, 4],     [-3, -2, -1, 1, 3, 5, 6]],
    79: ['Gold',       'Au', [3],        [-3, -2, -1, 1, 2, 5]],
    80: ['Mercury',    'Hg', [1, 2],     [-2, 4]],  # +4  doi:10.1002/anie.200703710
    81: ['Thallium',   'Tl', [1, 3],     [-5, -2, -1, 2]],
    82: ['Lead',       'Pb', [2, 4],     [-4, -2, -1, 1, 3]],
    83: ['Bismuth',    'Bi', [3],        [-3, -2, -1, 1, 2, 4, 5]],
    84: ['Polonium',   'Po', [-2, 2, 4], [5, 6]],
    85: ['Astatine',   'At', [-1, 1],    [3, 5, 7]],
    86: ['Radon',      'Rn', [2],        [6]],
    87: ['Francium',   'Fr', [1],        []],
    88: ['Radium',     'Ra', [2],        []],
    89: ['Actinium',   'Ac', [3],        []],
    90: ['Thorium',        'Th', [4],    [1, 2, 3]],
    91: ['Protactinium',   'Pa', [5],    [3, 4]],
    92: ['Uranium',        'U',  [6],    [1, 2, 3, 4, 5]],
    93: ['Neptunium',      'Np', [5],    [2, 3, 4, 6, 7]],
    94: ['Plutonium',      'Pu', [4],    [2, 3, 5, 6, 7]],
    95: ['Americium',      'Am', [3],    [2, 4, 5, 6, 7]],
    96: ['Curium',         'Cm', [3],    [4, 6]],
    97: ['Berkelium',      'Bk', [3],    [4]],
    98: ['Californium',    'Cf', [3],    [2, 4]],
    99: ['Einsteinium',    'Es', [3],    [2, 4]],
    100: ['Fermium',       'Fm', [3],    [2]],
    101: ['Mendelevium',   'Md', [3],    [2]],
    102: ['Nobelium',      'No', [2],    [3]],
    103: ['Lawrencium',    'Lr', [3],    []],
    104: ['Rutherfordium', 'Rf', [4],    []],
    105: ['Dubnium',       'Db', [5],    []],
    106: ['Seaborgium',    'Sg', [6],    []],
    107: ['Bohrium',       'Bh', [7],    []],
    108: ['Hassium',       'Hs', [8],    []],
    109: ['Meitnerium',    'Mt', [],     []],
    110: ['Darmstadtium',  'Ds', [],     []],
    111: ['Roentgenium',   'Rg', [],     []],
    112: ['Copernicium',   'Cn', [2],    []],
    113: ['Nihonium',      'Nh', [],     []],
    114: ['Flerovium',     'Fl', [],     []],
    115: ['Moscovium',     'Mc', [],     []],
    116: ['Livermorium',   'Lv', [],     []],
    117: ['Tennessine',    'Ts', [],     []],
    118: ['Oganesson',     'Og', [],     []],
}

node_matrix_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

edge_matrix_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


NUM_ATOMS = len(list(X_MAP['atomic_num']))
NUM_CHIRALITIES = len(list(X_MAP['chirality']))
NUM_DEGREES = len(list(X_MAP['degree']))
NUM_FORMAL_CHARGES = len(list(X_MAP['formal_charge']))
NUM_HS = len(list(X_MAP['num_hs']))
NUM_RADICAL_ELECTRONS = len(list(X_MAP['num_radical_electrons']))
NUM_HYBRIDIZATION = len(list(X_MAP['hybridization']))
NUM_AROMATIC = len(list(X_MAP['is_aromatic'])) # 2
NUM_INRING = len(list(X_MAP['is_in_ring'])) # 2

NUM_BOND_TYPES = len(list(E_MAP['bond_type']))
NUM_STEREO = len(list(E_MAP['stereo']))
NUM_CONJUGATED = len(list(E_MAP['is_conjugated'])) # 2

NODE_SHUFFLE_DECODER_DIMENSION = 25

BEST_PARAMETERS = {
    "batch_size": [128],
    "learning_rate": [0.01],
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
    "model_embedding_size": [1024],
    "model_attention_heads": [6],
    "model_layers": [8],
    "model_dropout_rate": [0.2],
    "model_top_k_ratio": [0.5],
    "model_top_k_every_n": [1],
    "model_dense_neurons": [256]
}

OPTIM_DICT = {
    "adam": torch.optim.Adam
}

SCHEDULER_DICT = {
    "plateau":torch.optim.lr_scheduler.ReduceLROnPlateau
}