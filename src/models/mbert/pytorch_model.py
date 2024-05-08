import csv
import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.data.data_config import Config
from src.helpers.logging_helper import setup_logging
from src.helpers.seed_helper import initialize_gpu_seed
from src.helpers.path_helper import *

from src.data.dataset import SwissTextDataset
from src.models.mbert.optimizer import build_optimizer
from src.models.mbert.config import write_config_to_file
sys.path.append(os.getcwd())

setup_logging()


class PyTorchModel:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = SwissTextDataset.create_instance(dataset=self.args.dataset, model_name=self.args.model_name,
                                                        use_val=self.args.use_val, seed=self.args.seed,
                                                        max_seq_length=self.args.max_seq_length,
                                                        do_lower_case=self.args.do_lower_case,
                                                        train_frac = self.args.train_frac, 
                                                        no_stopword_removal = self.args.no_stopword_removal)
    

        self.seed = self.args.seed
        self.model_seed = self.args.model_seed
        self.use_val = self.args.use_val

        self.model_config = Config.MODELS[self.args.model_name]

        self.network = self._get_pretrained_network(self.args.model_name, self.model_config.pretrained_model) 
        self.device, _ = initialize_gpu_seed(self.model_seed)
        self.network.to(self.device)

        self._setup_data_loaders()
        self._setup_optimizer()
        if self.test_data_loader is not None:
            self._reset_prediction_buffer()


    @staticmethod
    def load_from_args(args):
        initialize_gpu_seed(args.model_seed)

        checkpoint_suffix = '__epoch' + str(args.epoch)
        if args.epoch == 0:
            checkpoint_suffix += '__zeroshot'

        file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
        checkpoint_path = experiment_file_path(args.experiment_name, file_name)

        model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)
        return model

    @staticmethod
    def load_from_checkpoint(args, model_path: str = None):
        model = PyTorchModel(args)
        if model_path:
            model.load(model_path)
        else:
            logging.info('No :model_path provided, only loading base class.')
        return model

    def save(self, suffix: str = ""):
        file_name = "".join([self.args.model_name, suffix, '.pt'])
        model_path = experiment_file_path(self.args.experiment_name, file_name)

        if file_exists_or_create(model_path):
            raise ValueError(f'Checkpoint already exists at {model_path}')

        torch.save(self.network.state_dict(), model_path)
        logging.info(f"\tSuccessfully saved checkpoint at {model_path}")

        config_path = experiment_config_path(self.args.experiment_name)
        if not os.path.isfile(config_path):
            write_config_to_file(self.args)

    def load(self, model_path):
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))

    def _get_pretrained_network(self, model_name, pretrained_model):
        config_class = Config.MODELS[model_name].model_config
        model_class = Config.MODELS[model_name].model_class

        config = config_class.from_pretrained(pretrained_model)
        # Change the number of labels
        config.num_labels = self.dataset.num_labels
        try:
            network = model_class.from_pretrained(pretrained_model, config=config)
        except RuntimeError:
            network = model_class.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)

        # (Log)SoftmaxLayer to get softmax probabilities as output of our network, rather than logits
        if self.args.use_softmax_layer:
            new_clf = nn.Sequential(
                network.classifier,
                nn.LogSoftmax(dim=1),
            )
            network.classifier = new_clf

        return network
    
    def predict(self, batch_tuple):
        # Move all tensors in the batch tuple to the device (=GPU if available)
        batch = tuple(batch_tensor.to(self.device) for batch_tensor in batch_tuple)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[3]
        }

        model_names = ['distilbert', 'qlora-mistral']
        if self.args.model_name not in model_names:
            inputs['token_type_ids'] = batch[2] if self.args.model_name in ['bert', 'mbert'] else None

        outputs = self.network(**inputs)
        return outputs, inputs
    
    
    def train(self):
        global_step = 0
        total_loss, prev_epoch_loss, prev_loss = 0.0, 0.0, 0.0
        self.epoch_losses, self.train_accuracies, self.test_accuracies = [], [], []

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
                output = outputs[1]

                #  We use a (Log)SoftmaxLayer in addition to the appropriate loss function
                # (see https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
                #
                if self.args.use_softmax_layer:
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
            self.epoch_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

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
        
        # Save the training losses and accuracies
        file_name = "".join([self.args.model_name, '__training_log.csv']) 
        log_path = experiment_file_path(self.args.experiment_name, file_name)
        file_exists_or_create(log_path)
        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            if len(self.test_accuracies) == 0:
                writer.writerow(['loss', 'train_accuracy'])
                writer.writerows(zip(self.epoch_losses, self.train_accuracies))

            else:   
                writer.writerow(['loss', 'train_accuracy', 'test_accuracy'])
                writer.writerows(zip(self.epoch_losses, self.train_accuracies, self.test_accuracies))


    def test(self, epoch: int = 0, global_step: int = 0):
        self._reset_prediction_buffer()
        total_loss, prev_loss = 0.0, 0.0
        sample_count, sample_correct = 0, 0

        for step, batch_tuple in tqdm(enumerate(self.test_data_loader), desc=f'[TESTING] Running epoch {epoch} ...',
                                      total=len(self.test_data_loader)):
            self.network.eval()
            outputs, inputs = self.predict(batch_tuple)

            loss = outputs[0]
            output = outputs[1]
            total_loss += loss.item()

            # Calculate how many were correct
            predictions = torch.argmax(output, axis=1)

            if self.args.use_softmax_layer:
                prediction_proba, _ = torch.max(torch.exp(output)[:, predictions], dim=1)
            else:
                prediction_proba, _ = torch.max(torch.nn.functional.softmax(output, dim=2)[:, predictions], dim=1)

            self.log_predictions(inputs['labels'], predictions, prediction_proba, step=step)

            sample_count += len(predictions)
            sample_correct += (predictions == inputs['labels'].squeeze()).detach().cpu().numpy().sum()

        test_loss = round(total_loss / len(self.test_data_loader), 4)
        test_acc = round(sample_correct / sample_count, 4)
        if hasattr(self, 'test_accuracies'):
            self.test_accuracies.append(test_acc)

        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tTest Loss: {test_loss}\tTest Accuracy: {test_acc}")
        self.save_test_predictions(epoch)

    def validate(self, epoch: int = 0, global_step: int = 0):
        if self.val_data_loader is None:
            logging.info(f'No validation data loader for this dataset -> skipping validation step.')
            return
        self._reset_prediction_buffer(is_test=False)
        total_loss, prev_loss = 0.0, 0.0
        sample_count, sample_correct = 0, 0

        for step, batch_tuple in tqdm(enumerate(self.val_data_loader), desc=f'[VALIDATE] Running epoch {epoch} ...',
                                      total=len(self.val_data_loader)):
            self.network.eval()
            outputs, inputs = self.predict(batch_tuple)

            loss = outputs[0]
            output = outputs[1]
            total_loss += loss.item()

            # Calculate how many were correct
            predictions = torch.argmax(output, axis=1)

            if self.args.use_softmax_layer:
                prediction_proba, _ = torch.max(torch.exp(output)[:, predictions], dim=1)
            else:
                prediction_proba, _ = torch.max(torch.nn.functional.softmax(output, dim=2)[:, predictions], dim=1)

            self.log_predictions(inputs['labels'], predictions, prediction_proba, step=step, is_test=False)

            sample_count += len(predictions)
            sample_correct += (predictions == inputs['labels'].squeeze()).detach().cpu().numpy().sum()

        val_loss = round(total_loss / len(self.test_data_loader), 4)
        val_acc = round(sample_correct / sample_count, 4)

        logging.info(f"[Epoch {epoch}/{self.args.num_epochs}]\tVal Loss: {val_loss}\tVal Accuracy: {val_acc}")

    def ensemble_test(self,):
        self._reset_prediction_buffer()
        outputs_list = []
        labels_list = []

        for step, batch_tuple in tqdm(enumerate(self.test_data_loader), desc='[TESTING] Running ensemble test for {} ...'.format(self.args.model_name),
                                      total=len(self.test_data_loader)):
            self.network.eval()
            outputs, inputs = self.predict(batch_tuple)

            output = outputs[1]

            outputs_list.append(torch.argmax(output, axis=1))
            labels_list.append(inputs['labels'])

        # Concatenate all the outputs and labels
        outputs_list = torch.cat(outputs_list, dim=0)
        labels_list = torch.cat(labels_list, dim=0)

        return outputs_list, labels_list



    def test_loader(self, loader):
        all_predictions = []

        for step, batch_tuple in tqdm(enumerate(loader), desc='[TESTING] Custom Data Loader ...',
                                      total=len(loader)):
            self.network.eval()
            outputs, _, _ = self.predict(batch_tuple)
            output = outputs[1]

            predictions = torch.argmax(output, axis=1)

            if self.args.use_softmax_layer:
                prediction_proba = torch.exp(output)[:, 1, predictions]
            else:
                prediction_proba = torch.nn.functional.softmax(output, dim=2)[:, 1, predictions]

            all_predictions.extend(prediction_proba.cpu().detach().numpy())
        return all_predictions


    def log_predictions(self, labels, predictions, prediction_proba, step=0, is_test=True):
        def tensor_to_list(tensor_data):
            return tensor_data.detach().cpu().numpy().reshape(-1).tolist()

        batch_size = self.test_data_loader.batch_size
        num_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)

        start_idx = step * batch_size
        end_idx = np.min([((step + 1) * batch_size), num_samples])

        # Save the IDs, labels and predictions in the buffer
        self.prediction_buffer['labels'][start_idx:end_idx] = tensor_to_list(labels)
        self.prediction_buffer['predictions'][start_idx:end_idx] = tensor_to_list(predictions)
        self.prediction_buffer['prediction_proba'][start_idx:end_idx] = tensor_to_list(prediction_proba)

    # Method for saving the values in the prediction_buffer.
    # Note that calling this method also resets the buffer
    def save_test_predictions(self, epoch):
        file_name = "".join([self.args.model_name, '__prediction_log__ep', str(epoch), '.csv'])
        log_path = experiment_file_path(self.args.experiment_name, file_name)

        file_exists_or_create(log_path)
        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.prediction_buffer.keys())
            writer.writerows(zip(*self.prediction_buffer.values()))

    def _reset_prediction_buffer(self, is_test=True):
        num_test_samples = len(self.test_data_loader.dataset) if is_test else len(self.val_data_loader.dataset)
        self.prediction_buffer = {
            'labels': np.zeros(num_test_samples, dtype=int),
            'predictions': np.zeros(num_test_samples, dtype=int),
            'prediction_proba': np.zeros(num_test_samples, dtype=float)
        }

    def _setup_optimizer(self):
        num_train_steps = len(self.train_data_loader) * self.args.num_epochs
        optimizer, scheduler = build_optimizer(self.network,
                                               num_train_steps,
                                               self.args.learning_rate,
                                               self.args.adam_eps,
                                               self.args.warmup_steps,
                                               self.args.weight_decay)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _setup_data_loaders(self):
        train_data_loader, test_data_loader, val_data_loader = self.dataset.get_data_loaders(batch_size=self.args.batch_size)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader
