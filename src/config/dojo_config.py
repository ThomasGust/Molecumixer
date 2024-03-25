import json


class DojoConfig:

    def __init__(self, epochs,
                       batch_size,
                       learning_rate,
                       weight_decay,
                       sgd_momentum,
                       scheduler_gamma, 
                       pos_weight, 
                       model_embedding_size,
                       model_attention_heads,
                       model_layers,
                       model_dropout_rate,
                       model_top_k_ratio,
                       model_top_k_every_n,
                       model_dense_neurons,
                       dataloader_root,
                       optimizer,
                       scheduler,
                       scheduler_patience):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sgd_momentum = sgd_momentum
        self.scheduler_gamma = scheduler_gamma
        self.pos_weight = pos_weight
        self.model_embedding_size = model_embedding_size
        self.model_attention_heads = model_attention_heads
        self.model_layers = model_layers
        self.model_dropout_rate = model_dropout_rate
        self.model_top_k_ratio = model_top_k_ratio
        self.model_top_k_every_n = model_top_k_every_n
        self.model_dense_neurons = model_dense_neurons
        self.dataloader_root = dataloader_root
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
    
    def as_dict(self):
        d = {
            "epochs":self.epochs,
            "batch_size":self.batch_size,
            "learning_rate":self.learning_rate,
            "sgd_momentum":self.sgd_momentum,
            "scheduler_gamma":self.scheduler_gamma,
            "pos_weight":self.pos_weight,
            "model_embedding_size":self.model_embedding_size,
            "model_attention_heads":self.model_attention_heads,
            "model_layers":self.model_layers,
            "model_dropout_rate":self.model_dropout_rate,
            "model_top_k_ratio":self.model_top_k_ratio,
            "model_top_k_every_n":self.model_top_k_every_n,
            "model_dense_neurons":self.model_dense_neurons,
            "dataloader_root":self.dataloader_root,
            "optimizer":self.optimizer,
            "scheduler":self.scheduler,
            "scheduler_patience":self.scheduler_patience
        }
        return d

    def register_json(self, sp):
        d = self.as_dict()
        with open(sp, "w") as f:
            json.dump(d, f)
        return d
    
    def from_json(self, sp):
        with open(sp, "rb") as f:
            j = json.load(f)

        self.epochs = j['epochs']
        self.batch_size = j['batch_size']
        self.learning_rate = j['learning_rate']
        self.sgd_momentum = j['sgd_momentum']
        self.scheduler_gamma = j['scheduler_gamma']
        self.pos_weight = j['pos_weight']
        self.model_embedding_size = j['model_embedding_size']
        self.model_attention_heads = j['model_attention_heads']
        self.model_layers = j['model_layers']
        self.model_dropout_rate = j['model_dropout_rate']
        self.model_top_k_ratio = j['model_top_k_ratio']
        self.model_top_k_every_n = j['model_top_k_every_n']
        self.model_dense_neurons = j['model_dense_neurons']
        self.dataloader_root = j['dataloader_root']
        self.optimizer = j['optimizer']
        self.scheduler = j['scheduler']
        self.scheduler_patience = j['scheduler_patience']

def load_dojo_config(p):
    blank_c = DojoConfig(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    blank_c.from_json(p)
    return blank_c

if __name__ == "__main__":
    config = DojoConfig(epochs=50,
                        batch_size=32,
                        learning_rate=1e-3,
                        weight_decay=1e-5,
                        sgd_momentum=0.8,
                        scheduler_gamma=0.8,
                        pos_weight=1.3,
                        model_embedding_size=1024,
                        model_attention_heads=6,
                        model_layers=8,
                        model_dropout_rate=0.2,
                        model_top_k_ratio=0.5,
                        model_top_k_every_n=1,
                        model_dense_neurons=256,
                        dataloader_root="data\\dataloader",
                        optimizer="adam",
                        scheduler="plateau",
                        scheduler_patience=3)
    config.register_json(sp="config.tc")
