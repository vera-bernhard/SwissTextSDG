from ..mbert.pytorch_model import PytorchModel
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


class QloraModel(PytorchModel):
    
    def __init__(self, args):
        super().__init__(args)
        
        self.network = self._get_pretrained_networks(args.model_name, args.pretrained_model)
        self.network = self._prepare_for_peft(self.network)                
    # def train():
    #     pass
        
        
    # def predict():
    #     pass
    
    def _get_pretrained_networks(self, model_name, pretrained_model):
        # Loading in 4-bits & FN4 quantization as in QLORA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= 'nf4',
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= True,
        )
        
        config_class = self.MODELS[model_name].model_config
        model_name = self.MODELS[model_name].pretrained_model
        
        network = config_class.from_pretrained(
            model_name,
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