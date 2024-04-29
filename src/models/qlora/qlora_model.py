from src.models.mbert.pytorch_model import PyTorchModel
from transformers import (AutoTokenizer,
                          MistralForSequenceClassification, 
                          BitsAndBytesConfig, 
                          Trainer, 
                          TrainingArguments)
from peft import (LoraConfig, 
                  PeftConfig, 
                  PeftModel, 
                  get_peft_model,
                  prepare_model_for_kbit_training)
import torch
from src.data.dataset import SwissTextDataset
from src.data.data_config import Config
from src.helpers.seed_helper import initialize_gpu_seed


from huggingface_hub import login
from config import *


class QloraModel(PyTorchModel):
    
    def __init__(self, args):
        login(token=HUGGINGFACE_TOKEN)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = SwissTextDataset.create_instance(dataset=self.args.dataset, model_name=self.args.model_name,
                                                        use_val=self.args.use_val, seed=self.args.seed,
                                                        max_seq_length=self.args.max_seq_length,
                                                        do_lower_case=self.args.do_lower_case,
                                                        train_frac = self.args.train_frac,)
    

        self.seed = self.args.seed
        self.model_seed = self.args.model_seed
        self.use_val = self.args.use_val

        self.model_config = Config.MODELS[self.args.model_name]

        self.network = self._get_pretrained_networks(args.model_name, self.model_config.pretrained_model)
        self.network = self._prepare_for_peft(self.network)   
        
        self.device, _ = initialize_gpu_seed(self.model_seed)
        self.network.to(self.device)

        self._setup_data_loaders()
        self._setup_optimizer()
        if self.test_data_loader is not None:
            self._reset_prediction_buffer()


    # def load_from_checkpoint(args, model_path: str = None):
        # pass
    
    # def load(self, model_path):
    
    def _get_pretrained_networks(self, model_name, pretrained_model):
        # Loading in 4-bits & FN4 quantization as in QLORA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= 'nf4',
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= True,
        )
        
        pretrained_model = Config.MODELS[model_name].pretrained_model
        model_class = Config.MODELS[model_name].model_class   
        network = model_class.from_pretrained(
            pretrained_model,
            num_labels = self.dataset.num_labels,
            quantization_config = bnb_config,
            device_map = self.device,
            trust_remote_code=True
        )
        return network
    
    def _prepare_for_peft(self, network):
        network = prepare_model_for_kbit_training(network)
        peft_config = LoraConfig(
            lora_alpha=16, # Scaling of Adapaters
            lora_dropout=0.1, 
            r=2, # rank of low-rank adaption
            bias='none',
            task_type='SEQ_CLS',
            target_modules=['q_proj', 'v_proj']
        )
        # add adapters
        return get_peft_model(network, peft_config)