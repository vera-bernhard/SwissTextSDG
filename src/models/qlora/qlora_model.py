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
from src.helpers.path_helper import *
from src.models.mbert.config import write_config_to_file

from tqdm.auto import tqdm

from huggingface_hub import login
from config import *
import torch.nn as nn
from src.helpers.logging_helper import setup_logging
import logging
setup_logging()


class QloraModel(PyTorchModel):
    
    def __init__(self, args, load=False):
        login(token=HUGGINGFACE_TOKEN)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = SwissTextDataset.create_instance(dataset=self.args.dataset, model_name=self.args.model_name,
                                                        use_val=self.args.use_val, seed=self.args.seed,
                                                        max_seq_length=self.args.max_seq_length,
                                                        do_lower_case=self.args.do_lower_case,
                                                        train_frac = self.args.train_frac,
                                                        no_stopword_removal=self.args.no_stopword_removal)
    
        
        self.seed = self.args.seed
        self.model_seed = self.args.model_seed
        self.use_val = self.args.use_val

        self.model_config = Config.MODELS[self.args.model_name]            
        self.network = self._get_pretrained_network(self.model_config.pretrained_model)

        if load == False:
            self.network = self._prepare_for_peft(self.network)   
            self._prepare_for_training()

    def _prepare_for_training(self, load=False):
        self.device, _ = initialize_gpu_seed(self.model_seed)
        if not load: 
            self.network.to(self.device)

        self._setup_data_loaders()
        self._setup_optimizer()
        if self.test_data_loader is not None:
            self._reset_prediction_buffer()
            
    @staticmethod
    def load_from_checkpoint(args, model_path: str = None):
        model = QloraModel(args, load=True)
        if model_path:
            model.load(model_path)
        else:
            logging.info('No :model_path provided, only loading base class.')
        return model


    def load(self, model_path: str, trainable: bool = True):
        self.network = PeftModel.from_pretrained(
            self.network,
            model_path,
            is_trainable=trainable
        )
        self._prepare_for_training(load=True)
            
    def save(self, suffix: str = ''): 
        file_name = "".join([self.args.model_name, suffix])
        model_path = experiment_file_path(self.args.experiment_name, file_name)
        self.network.save_pretrained(model_path)
        logging.info(f"\tSuccessfully saved checkpoint at {model_path}")

        config_path = experiment_config_path(self.args.experiment_name)
        if not os.path.isfile(config_path):
            write_config_to_file(self.args)
                
    def train(self):
        global_step = 0
        total_loss, prev_epoch_loss, prev_loss = 0.0, 0.0, 0.0

        # Run zero-shot on test set
        if self.test_data_loader is not None:
            self.test()
            if self.args.save_model:
                self.save(suffix='__epoch0__zeroshot')

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc=f'Training for {self.args.num_epochs} epochs ...'):
            sample_count, sample_correct = 0, 0

            for step, batch_tuple in tqdm(enumerate(self.train_data_loader),
                                          desc=f'[TRAINING] Running epoch {epoch}/{self.args.num_epochs} ...',
                                          total=len(self.train_data_loader)):
                self.network.train()
                outputs, inputs = self.predict(batch_tuple)

                loss = outputs[0]
                output = torch.nn.functional.softmax(outputs[1], dim=1)
                #  We use a (Log)SoftmaxLayer in addition to the appropriate loss function
                # (see https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
                #
                loss_fn = nn.NLLLoss()
                loss = loss_fn(output, inputs['labels'].view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

                total_loss += loss.item()

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.network.zero_grad()

                # Calculate how many were correct
                predictions = torch.argmax(output, axis=1)
                batch_count = len(predictions)
                batch_correct = (predictions == inputs['labels'].squeeze()).detach().cpu().numpy().sum()

                sample_count += batch_count
                sample_correct += batch_correct
                prev_loss = total_loss
                global_step += 1

            train_loss = round((total_loss - prev_epoch_loss) / len(self.train_data_loader), 4)
            train_acc = round(sample_correct / sample_count, 4)

            logging.info(
                f"[Epoch {epoch}/{self.args.num_epochs}]\tTrain Loss: {train_loss}\tTrain Accuracy: {train_acc}")

            prev_epoch_loss = total_loss

            if self.args.save_model:
                self.save(suffix=f'__epoch{epoch}')

            # Run test+val set after each epoch
            if self.test_data_loader is not None:
                self.test(epoch)
            if self.use_val:
                self.validate(epoch)

    def test(self, epoch: int = 0, global_step: int = 0):
        self._reset_prediction_buffer()
        total_loss, prev_loss = 0.0, 0.0
        sample_count, sample_correct = 0, 0

        for step, batch_tuple in tqdm(enumerate(self.test_data_loader), desc=f'[TESTING] Running epoch {epoch} ...',
                                      total=len(self.test_data_loader)):
            self.network.eval()
            outputs, inputs = self.predict(batch_tuple)

            loss = outputs[0]
            output = torch.nn.functional.softmax(outputs[1], dim=1)
            total_loss += loss.item()

            # Calculate how many were correct
            predictions = torch.argmax(output, axis=1)

            prediction_proba = torch.gather(output, 1, predictions.unsqueeze(1))

            self.log_predictions(inputs['labels'], predictions, prediction_proba, step=step)

            sample_count += len(predictions)
            sample_correct += (predictions == inputs['labels'].squeeze()).detach().cpu().numpy().sum()

        test_loss = round(total_loss / len(self.test_data_loader), 4)
        test_acc = round(sample_correct / sample_count, 4)

        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tTest Loss: {test_loss}\tTest Accuracy: {test_acc}")
        self.save_test_predictions(epoch)

    def _get_pretrained_network(self, pretrained_model):
        # Loading in 4-bits & FN4 quantization as in QLORA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= 'nf4',
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= True,
        )
        
        model_class = self.model_config.model_class  
        network = model_class.from_pretrained(
            pretrained_model,
            num_labels = self.dataset.num_labels,
            quantization_config = bnb_config,
            device_map = self.device,
        )
        network.config.pad_token_id = self.dataset.tokenizer.tokenizer.pad_token_id
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