from section_1_models import BasicNet,BasicNetSkipCon,FourierNet
from section_1_models import ModelSim
from pytorch_lightning.callbacks import EarlyStopping
from section_1_models import SimEvalCallback
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import EarlyStopping
from section_1_models import SimEvalCallback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from section_1_models import DataModule
import fire


output_config = {

    "results_root_dir": "./results",
    "save_every": 10,
    "logs_dir": "logs",
    "name_experiment": None,


}


input_config = {
    "skip_steps":6,
    "max_data":None,
    "data_dir":"./data/AC",
    "gpus":-1,
}

model_config = {

    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 1e-4,
    "lr": 5*1e-4,
    "normalization": False,
    "n_blocks": 6,
    "layers_per_block": 3,
    "channels": 70,
    "name_model": "BasicCNN",
    "skip_con_weight":0.1,
    "modes_fourier": 16,
    "width_fourier": 60,
}





def train(input_config, output_config, model_config):

    ic, oc, mc = input_config, output_config, model_config

    results_root_dir, logs_dir, name_experiment = oc["results_root_dir"], oc["logs_dir"], oc["name_experiment"]

    if name_experiment == None:
        name_experiment = "experiment_" + mc["name_model"]

    results_dir = os.path.join(results_root_dir, name_experiment)

    models_checkpoints_dir = os.path.join(logs_dir, "models_checkpoints")


    print(""" #### READING AND PREPARING DATA skip_steps {}, data_dir {} ####""".format(ic["skip_steps"], ic["data_dir"]))

    datamod = DataModule(ic["data_dir"], max_data = ic["max_data"])

    datamod.prepare_data()

    datamod.setup(skip_steps = ic["skip_steps"])


    blocks = [mc["layers_per_block"]]*mc["n_blocks"]
    name = mc["name_model"]

    print("""#### Running {} with model {},
            blocks {}, hidden_channels {},
            skip_steps {}, max_epochs {}####""".format(name_experiment, name, blocks, mc["channels"], ic["skip_steps"], mc["max_epochs"]))

    if name == "BasicCNN":

        model = ModelSim(BasicNet,1,mc["channels"], 1,
                 blocks= blocks,
                 results_dir = results_dir,
                 lr = mc["lr"],normalization = mc["normalization"])

    elif name == "CNNskipcon":


        model = ModelSim(BasicNetSkipCon,1,mc["channels"], 1,
                 blocks= blocks,
                 skip_con_weight = mc["skip_con_weight"],
                 results_dir = results_dir,
                 lr = mc["lr"],normalization = mc["normalization"])

    elif name == "FourierNet":

        model = ModelSim(FourierNet,mc["modes_fourier"],mc["width_fourier"],
                 results_dir = results_dir, lr = mc["lr"])

    else:
        raise(ValueError("Available models {}".format(["BasicCNN", "CNNskipcon","FourierNet"])))

    logger_csv = CSVLogger(logs_dir, name= name_experiment)
    logger_tensorboard = TensorBoardLogger(logs_dir, name= name_experiment)


    callback = SimEvalCallback(datamod, results_dir,save_every = 10)
    early_stopping = EarlyStopping('val_loss', patience = 10, min_delta = 1e-4)

    trainer = pl.Trainer(max_epochs = mc["max_epochs"], callbacks = [early_stopping, callback],gpus=ic["gpus"],flush_logs_every_n_steps = 20, log_every_n_steps= 20,
                    logger = [logger_csv,logger_tensorboard],default_root_dir = models_checkpoints_dir)

    trainer.fit(model, datamod.train_dataloader(), datamod.val_dataloader())













class MainTrain():

    """
    -----------------
    train parameters:
    -----------------


    -------------------
    "results_root_dir": "./results",
    "save_every": 10,
    "logs_dir": "logs",
    "name_experiment": None,


    -----------------
    "skip_steps":6,
    "max_data":None,
    "data_dir":"./data/AC",



    ---------------------
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 1e-4,
    "lr": 5*1e-4,
    "normalization": False,
    "n_blocks": 6,
    "layers_per_block": 3,
    "channels": 70,
    "name_model": "BasicCNN",
    "skip_con_weight":0.1,


    """

    def train(self,**kwargs):

        for key,val in kwargs.items():

            for _dict in [input_config, output_config, model_config]:
                if key in _dict:
                    _dict[key] = val

        train(input_config, output_config, model_config)


    def run_predefined(self):


        names = ["CNNskipcon","BasicCNN"]

        for name in names:

            model_config["name_model"] = name

            train(input_config, output_config, model_config)






if __name__ == "__main__":
    fire.Fire(MainTrain)
