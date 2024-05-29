from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoTokenizer, LogitsProcessorList
import torch
import numpy as np
import pandas as pd
import json, argparse, os, sys
from tqdm import tqdm
import logging

sys.path.append("../")
from torch.utils.data import DataLoader
from utils.instructBLIPdataset import CustomVisionDataset
from utils.utilities import (
    set_logger,
    split_,
    split_data,
    load_tags_data,
    load_imageclef_data
)

from dmm import DMM
from dmm_logits import DMMLogits

class InstructBLIP:
    def __init__(self, config_path):
        # Load configurations from a JSON file
        self.load_config(config_path)

        n_gpu = self.config.get('n_gpu', 1)
        self.device = f"cuda:{str(self.config['cuda_nr'])}" if n_gpu > 0 else "cpu"

        set_logger("../snapshots/logs/", "train.log", self.config["logging"]["print_on_screen"])

    def load_config(self, config_path):
        """ Load all arguments from a JSON configuration file """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def parse_args(self) -> None:
        """ Dummy method to keep consistency if needed elsewhere """
        pass

    # Creating the training function. This will be called in the main function. It is run depending on the epoch value.
    # The model is put into train mode and then we wnumerate over the training loader and passed to the defined network
    def train(self, epoch, instruction_, model, processor, device, loader, optimizer):
        self.model.train() # switch into training mode
        running_loss = 0 # define loss
        batch_counter = 0 # define batch counter

        # and start the training loop!
        for _, data in tqdm(enumerate(loader, 0)):
            image = data[0]
            caption = data[1]
            ids = data[2]

            batch_counter += 1

            instruction = [instruction_ for i in range(len(caption))]

            inputs = processor(images=image, text=instruction, return_tensors="pt")
            inputs = inputs.to(device)

            labels = processor.tokenizer(caption, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
            labels["input_ids"] = torch.tensor([[-100 if x == processor.tokenizer.pad_token_id else x for x in labels["input_ids"][i].tolist()] for i in range(len(caption))])
            labels = torch.tensor(labels.input_ids).to(device)

            outputs = model(**inputs, labels = labels)

            loss = outputs.loss
            running_loss += loss.item()

            if _%500==0:
                logging.info(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        del inputs
        del labels
            

        epoch_loss = running_loss / batch_counter
        logging.info(f'Epoch: {epoch}, Loss:  {epoch_loss}, Batch Counter: {batch_counter}')


    def validate(self, epoch, instruction_, model, processor, device, loader):
        self.model.eval()
        predictions = []
        actuals = []
        val_batch_counter = 0
        running_val_loss = 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(loader, 0)):
                val_batch_counter += 1

                image = data[0]
                caption = data[1]
                ids = data[2]

                instruction = [instruction_ for i in range(len(caption))]

                inputs = processor(images=image, text=instruction, return_tensors="pt")
                inputs = inputs.to(device)

                labels = processor.tokenizer(caption, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
                labels["input_ids"] = torch.tensor([[-100 if x == processor.tokenizer.pad_token_id else x for x in labels["input_ids"][i].tolist()] for i in range(len(caption))])
                labels = torch.tensor(labels.input_ids).to(device)

                val_outputs = model(**inputs, labels = labels)

                val_loss = val_outputs.loss
                running_val_loss += val_loss.item()
            
            del inputs
            del labels
                

            epoch_val_loss = running_val_loss / val_batch_counter
            logging.info(f'Epoch: {epoch}, Validation Loss:  {epoch_val_loss}, Batch Counter: {val_batch_counter}')
            if (epoch_val_loss < self.best_loss):
                    self.best_loss = epoch_val_loss
                    self.early_stopping_counter = 0

                    # save model in order to retrieve at the end...
                    torch.save(model.state_dict(), self.config["BEST_MODEL_PATH"])
            else:
                    self.early_stopping_counter += 1

    

    
    def test(self, instruction_, model, processor, device, loader, hist_file_path, mmc_sim_file_path, tokenizer):

        model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for _, data in tqdm(enumerate(loader, 0)):

                image = data[0]
                caption = data[1]
                tags = data[2]

                instruction = [instruction_ for i in range(len(caption))]

                inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)

                generation_params = self.config["generation_params"]

                if self.config["dmmcs_params"]["do_dmmcs"]:

                    dmm = DMM(hist_train_file=hist_file_path,
                              mmc_sim_file=mmc_sim_file_path,
                              word_index_file=self.config["word_index_path"],
                              embedding_matrix_file=self.config["embedding_matrix_path"])

                    # instantiating a list of LogitsProcessor instances
                    # using our custom ABCLogits class
                    alpha = self.config["dmmcs_params"]["alpha"]
                    logits_processor = LogitsProcessorList([DMMLogits(dmm, tags, alpha, tokenizer)])

                    generation_params["logits_processor"] = logits_processor
                
                outputs = model.generate(**inputs, **generation_params)

                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

                if _%500==0:
                    logging.info(f'Completed inference on {str(_)} test instances.')
                    
                predictions.extend(generated_text)
                actuals.extend(caption)
        
        return predictions, actuals


    def main(self):

        # Set random seeds and deterministic pytorch for reproducibility
        torch.manual_seed(self.config["seed"]) # pytorch random seed
        np.random.seed(self.config["seed"]) # numpy random seed

        # load data in pandas dataframe form
        captions_train, captions_valid, captions_test, self._concepts_dict =    load_imageclef_data(
                                                                                  self.config["dataset_captions_path_train"],
                                                                                  self.config["dataset_captions_path_valid"],
                                                                                  self.config["dataset_captions_path_test"],
                                                                                  self.config["dataset_concepts_mapper"]
                                                                                )

        # load image ids individually
        train_ids, dev_ids, test_ids = split_data(captions_train, captions_valid, captions_test)

        tags_test = load_tags_data(self.config["dataset_concepts_path_test"])

        # load the captions individually
        train_labels, dev_labels, test_labels = split_(captions_train, captions_valid, captions_test)

        model = InstructBlipForConditionalGeneration.from_pretrained(self.config["train_params"]["checkpoint"])
        processor = InstructBlipProcessor.from_pretrained(self.config["train_params"]["checkpoint"])

        # define the instruction to be followed during training
        instruction = self.config["train_params"]["instruction"]

        self.model = model.to(self.device)
        self.processor = processor
        
        # Freeze some of the model's parameters in order to meet memory constraints -- adjust accordingly!
        for i, param in enumerate(self.model.vision_model.encoder.layers.parameters()):
            param.requires_grad = False

        for i, param in enumerate(self.model.language_model.encoder.parameters()):
            param.requires_grad = False
        
        c = 0
        for i, param in enumerate(self.model.language_model.decoder.parameters()):
            if i <= 334:
                param.requires_grad = False
            c += 1

        c2 = 0
        for i, param in enumerate(self.model.qformer.encoder.layer.parameters()):
            c2+=1
            if i <= 190:
                param.requires_grad = False

        true_, false_ = 0, 0
        for i, param in enumerate(self.model.parameters()):   
            g = param.requires_grad
            if (g):
                true_ += 1
            else:
                false_ += 1
            
        logging.info(f'Trainable model weights: {str(true_)}')
        logging.info(f'Frozen model weights: {str(false_)}')

        self.train_dataset = CustomVisionDataset(captions_train, train_ids, list(), self.config["dataset_images_path"], 'train')
        self.val_dataset = CustomVisionDataset(captions_valid, dev_ids, list(), self.config["dataset_images_path"], 'validation')
        self.test_dataset = CustomVisionDataset(captions_test, test_ids, tags_test, self.config["dataset_images_path"], 'test')

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.config["train_params"]["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': self.config["num_workers_train"]
        }

        val_params = {
            'batch_size': self.config["train_params"]["VALID_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': self.config["num_workers_val"]
        }

        test_params = {
            'batch_size': self.config["train_params"]["TEST_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': self.config["num_workers_test"]
        }


        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        training_loader = DataLoader(self.train_dataset, **train_params)
        val_loader = DataLoader(self.val_dataset, **val_params)
        test_loader = DataLoader(self.test_dataset, **test_params)

        logging.info(f"Running {self.config['train_params']['checkpoint']} on: {str(self.device)}")

        # Defining the optimizer that will be used to tune the weights of the network in the training session. 
        optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.config["train_params"]["lr"])

        logging.info('Initiating instruction-based fine-tuning for the model on our dataset...')

        self.best_loss = 1000000
        self.early_stopping_counter = 0
        if self.config["train_params"]["do_train"]:
            for epoch in tqdm(range(self.config["train_params"]["TRAIN_EPOCHS"])):
                self.train(epoch, instruction, self.model, self.processor, self.device, training_loader, optimizer)
                self.validate(epoch, instruction, self.model, self.processor, self.device, val_loader)

                # check for early stopping
                if ((self.early_stopping_counter >= self.config["train_params"]["early_stopping_threshold"]) or (epoch == (self.config["TRAIN_EPOCHS"] - 1))):

                    # delete existing instances in order to deal with potential memory issues
                    del self.model, model, optimizer

                    model = InstructBlipForConditionalGeneration.from_pretrained(self.config["train_params"]["checkpoint"])
                    state_dict = torch.load(self.config["BEST_MODEL_PATH"], map_location = 'cpu')
                    model.load_state_dict(state_dict)
                    model = model.to(self.device)
                    break

        logging.info('Now generating summaries on our fine tuned model for the test dataset and saving it in a dataframe')

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

        predictions, actuals = self.test(instruction, model, self.processor, self.device, test_loader, self.config["hist_file_path"], self.config["mmc_sim_file_path"], tokenizer)

        with open(self.config["RESULTS_PATH"], 'w') as out_test:
            for i, pred in enumerate(predictions):
                out_test.write(test_ids[i] + '|' + pred + '\n')
        logging.info('Results saved!')


if __name__ == '__main__':
    # define tokenizer

    parser = argparse.ArgumentParser(description="Script to run InstructBLIP with JSON config.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    # Instantiate InstructBLIP with the provided config file
    instruct_blip = InstructBLIP(args.config)

    # Run!
    instruct_blip.main()
