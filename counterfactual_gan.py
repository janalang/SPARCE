import numpy as np
import os
import torch
from torch import nn
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
from datetime import datetime
from loss_functions import *
from dataloader import *
from utils import *
from gan_models import *



class CounterfactualTimeGAN(nn.Module):
    """Implementation of a GAN architecture for the generation of counterfactual explanations. 
    Generator and discriminator are implemented as recurrent sequence classifiers.
    The generator either creates residuals (SPARCE) or entire sequences (GAN) based on input sequences.
    """

    def __init__(self):
        super().__init__()

    def build_model(self, args, device, num_features, bidirectional=True, hidden_dim_generator=32, layer_dim_generator=2, hidden_dim_discriminator=32, layer_dim_discriminator=2, classifier_model_name="bidirectional_lstm_classifier"):
        """Initialize all network elements (generator, discriminator and classifier) and define optimizers.

        Args:
            args (args): Input arguments.  
            device (device): 'gpu' or 'cpu'.
            num_features (int): Number of features in the input data.
            bidirectional (bool, optional): Indicates whether generator is a simple or bidirectional LSTM. Defaults to True.
            hidden_dim_generator (int, optional): Number of hidden neurons in the generator. Defaults to 32.
            layer_dim_generator (int, optional): Number of layers in the generator. Defaults to 2.
            hidden_dim_discriminator (int, optional): Number of hidden neurons in the discriminator. Defaults to 32.
            layer_dim_discriminator (int, optional): Number of layers in the discriminator. Defaults to 2.
            classifier_model_name (str, optional): Name of pretrained classification model. Defaults to "bidirectional_lstm_classifier".
        """

        self.device = device
        self.args = args
        generator_output_dim = num_features
        discrim_output_dim = 1 # true or false

        feature_mapping = get_feature_mapping(self.args.dataset)

        if not self.args.freeze_features:
            self.args.freeze_features = ['Center_x', 'Center_y'] if self.args.dataset == 'catching' else []

        freeze_indices = [feature_mapping.index(i) for i in self.args.freeze_features]

        # instantiate Generator model
        if bidirectional:
            generator = BidirectionalResidualGANLSTM(num_features, hidden_dim_generator, layer_dim_generator, generator_output_dim, freeze_indices)
        else:
            generator = ResidualGANLSTM(num_features, hidden_dim_generator, layer_dim_generator, generator_output_dim, freeze_indices)
            

        generator = generator.to(device)
        print(generator)
        print(f'Generator number of trainable parameters: {count_model_parameters(generator, trainable_only=True)}')

        discriminator = BidirectionalLSTM(num_features, hidden_dim_discriminator, layer_dim_discriminator, discrim_output_dim)

        discriminator = discriminator.to(device)
        print(discriminator)
        print(f'Discriminator number of trainable parameters: {count_model_parameters(discriminator, trainable_only=True)}')

        # load pretrained classification model
        classifier = load_model(args.dataset, classifier_model_name)
        print(f'Classifier number of trainable parameters: {count_model_parameters(classifier, trainable_only=True)}')

        # loss function
        loss_fn = nn.BCELoss()

        # optimizers
        generator_optim = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        discriminator_optim = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.generator = generator
        self.generator_optim = generator_optim
        self.discriminator = discriminator
        self.discriminator_optim = discriminator_optim
        self.classifier = classifier
        self.loss_fn = loss_fn

    def _compute_discriminator_loss(self, real_out, fake_out):
        """Compute the loss for the discriminator model. 
        Implemented as the averaged overall performance across real and fake samples.

        Args:
            real_out: Classifier predictions on real inputs.
            fake_out: Classifier predictions on fake input (counterfactuals).

        Returns:
            total_loss: Combined discriminator loss.
        """
        real_loss = self.loss_fn(real_out, torch.ones_like(real_out).to(self.device)) # compare predictions on real input to ones (all true)
        fake_loss = self.loss_fn(fake_out, torch.zeros_like(fake_out).to(self.device)) # compare predictions on fake input to zeros (all fake)
        total_loss = (real_loss + fake_loss) / 2 # overall performance across real and fake data
        return total_loss

    def _compute_classification_loss(self, out_cf, target_class):
        """Compute the loss between the predicted class of the counterfactual and the target class.

        Args:
            out_cf: Classifier predictions on counterfactuals.
            target_class: Desired target class for counterfactuals.

        Returns:
            dist_pred_target: Classification loss
        """
        crossentropy = torch.nn.CrossEntropyLoss(reduction="none")
        dist_pred_target = crossentropy(out_cf, target_class).to(self.device)
        return dist_pred_target

    def _compute_adversarial_loss(self, fake_out):
        """Compute the generator's adversarial loss specifying how close the predictions on fake sequences are to predictions on real sequences.

        Args:
            fake_out: Classifier predictions on counterfactuals.

        Returns:
            adv_loss: Adversarial loss.
        """
        # how close are the predictions on the fake sequences to predictions on real sequences?
        bce = torch.nn.BCELoss(reduction="none")
        adv_loss = bce(fake_out, torch.ones_like(fake_out).to(self.device)) # compare predictions on fake input to ones (all true) 
        return adv_loss

    def _compute_generator_loss(self, adv_loss, cls_loss, similarity_loss, sparsity_loss, jerk_loss):
        """Compute the combined weighted generator loss. Compute average loss over batch.

        Args:
            adv_loss: GAN-based adversarial loss.
            cls_loss: Classification loss.
            similarity_loss: Similarity loss (L1 norm) between counterfactual and query.
            sparsity_loss: Sparsity loss (L0 norm) between counterfactual and query.
            jerk_loss: Jerk loss, i.e. difference between residuals in subsequent time steps.

        Returns:
            batch_loss: Weighted averaged generator loss.
        """
        fake_loss = (self.args.lambda1 * adv_loss + self.args.lambda2 * cls_loss + self.args.lambda3 * similarity_loss + self.args.lambda4 * sparsity_loss + self.args.lambda5 * jerk_loss).to(self.device)
        batch_loss = torch.mean(fake_loss)
        return batch_loss

    def _train_step(self, real_input, real_labels, generator_input):
        """Train generator and discriminator.

        Args:
            real_input: Targets
            real_labels: Target labels
            generator_input: Queries

        """

        feature_mapping = get_feature_mapping(self.args.dataset)

        freeze_indices = [feature_mapping.index(i) for i in self.args.freeze_features]
        remaining_indices = [x for x in range(len(feature_mapping)) if x not in freeze_indices]

        if not feature_mapping:
            remaining_indices = range(0, real_input.shape[2])

        deltas = self.generator(generator_input)

        deltas_with_freeze = torch.zeros_like(generator_input)
        deltas_with_freeze[:,:,remaining_indices] = deltas
        cf = deltas_with_freeze + generator_input
        out_cf = self.classifier(cf).to(self.device)
        out_cf_class = torch.argmax(out_cf, dim=1)
        target_class = torch.argmax(real_labels, dim=1).to(self.device)

        cls_loss = self._compute_classification_loss(out_cf, target_class)
        similarity_loss = compute_similarity_loss(deltas)
        sparsity_loss = compute_sparsity_loss(deltas)
        jerk_loss = compute_jerk_loss(deltas, self.device)

        fake_out = self.discriminator(cf)
        adv_loss = self._compute_adversarial_loss(fake_out)
        generator_loss = self._compute_generator_loss(adv_loss, cls_loss, similarity_loss, sparsity_loss, jerk_loss)

        print(f'[Avg Classification Loss: {np.mean(cls_loss.detach().cpu().numpy()):.5f}] [Avg Adversarial Loss: {np.mean(adv_loss.detach().cpu().numpy()):.5f}] [Avg Similarity Loss: {np.mean(similarity_loss.detach().cpu().numpy()):.5f}] [Avg Sparsity Loss: {np.mean(sparsity_loss.detach().cpu().numpy()):.5f}] [Avg Jerk Loss: {np.mean(jerk_loss.detach().cpu().numpy()):.5f}]')

        self.generator_optim.zero_grad() # clear the gradients to avoid accumulating gradients
        generator_loss.backward() 
        self.generator_optim.step()

        real_out = self.discriminator(real_input)
        fake_out_no_grad = self.discriminator(cf.detach())
        discriminator_loss = self._compute_discriminator_loss(real_out, fake_out_no_grad)
        self.discriminator_optim.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optim.step()

        losses = {
        'Classification Loss': cls_loss.detach().cpu().numpy(),
        'Adversarial Loss': adv_loss.detach().cpu().numpy(),
        'Similarity Loss': similarity_loss.detach().cpu().numpy(),
        'Sparsity Loss': sparsity_loss.detach().cpu().numpy(),
        'Jerk Loss': jerk_loss.detach().cpu().numpy(),
        'Generator Loss': generator_loss.detach().cpu().item(),
        'Discriminator Loss': discriminator_loss.detach().cpu().item()
        }

        return cf, generator_loss, discriminator_loss, out_cf_class, losses


    def train(self, train_dl, generator_dl, max_samples, max_batches):
        """Train generator and discriminator over batches and epochs.

        Args:
            train_dl: Dataloader containing all target samples for training.
            generator_dl: Dataloader containing all query samples for training.
            max_samples: Maximum number of samples to be used.
            max_batches: Maximum number of batches to be used.
        """
        print("Starting to train the model...\n-------------------------------")
        train_loss_per_epoch_generator, train_loss_per_epoch_discriminator, losses_overall = [], [], []
        self.args.max_batches = max_batches
        num_batches = max_batches
        num_samples = max_samples

        print(f'Number of samples used: {max_samples}')
        print(f'Number of batches: {max_batches}')

        for epoch in range(self.args.num_epochs):
            print(f'Epoch: {epoch+1}\n-------------------------------')

            # switch to train mode (e.g. using dropout)
            self.generator.train()
            self.discriminator.train()

            train_generator_loss, train_discriminator_loss = 0, 0
            sequences_per_epoch, target_samples_per_epoch, query_samples_per_epoch, query_labels_per_epoch, out_cf_per_epoch, losses_per_epoch = [], [], [], [], [], []
            batch_count = 0
            num_samples = 0
            for data in zip(train_dl, generator_dl):

                X = data[0][0].to(self.device)
                y = data[0][1].to(self.device)

                generator_input = data[1][0].to(self.device)
                y_generator_input = data[1][1].to(self.device)

                batch_count += 1
                if batch_count > max_batches:
                    num_batches = max_batches
                    break

                current_batchsize = X.shape[0]
                num_samples += current_batchsize

                # take a training step
                generated_input, generator_loss, discriminator_loss, out_cf_classes, losses = self._train_step(real_input=X, real_labels=y, generator_input=generator_input)            
                print(f'[Epoch: {epoch+1}/{self.args.num_epochs}] [Batch: {batch_count}/{int(max_batches)}] [D Loss: {discriminator_loss}] [G Loss: {generator_loss}] \n')

                sequences_per_epoch.append(generated_input.detach())
                target_samples_per_epoch.append(X.detach())
                query_samples_per_epoch.append(generator_input.detach())
                query_labels_per_epoch.append(y_generator_input.detach())
                out_cf_per_epoch.append(out_cf_classes)
                losses_per_epoch.append(losses)

                train_generator_loss += generator_loss
                train_discriminator_loss += discriminator_loss


            train_loss_per_epoch_generator.append(train_generator_loss / num_batches)
            train_loss_per_epoch_discriminator.append(train_discriminator_loss / num_batches)
            losses_overall.append(losses_per_epoch)

        generated_sequences = []
        for batch in range(len(sequences_per_epoch)):
            for sample in range(len(sequences_per_epoch[batch])):
                generated_sequences.append(sequences_per_epoch[batch][sample].cpu().numpy())

        original_sequences = []
        for batch in range(len(target_samples_per_epoch)):
            for sample in range(len(target_samples_per_epoch[batch])):
                original_sequences.append(target_samples_per_epoch[batch][sample].cpu().numpy())

        generated_sequences = np.array(generated_sequences)
        original_sequences = np.array(original_sequences)


    def test(self, test_dl, generator_dl, max_samples, max_batches, testdoc):
        """Evaluate model on test samples.

        Args:
            test_dl: Dataloader containing all target samples for testing.
            generator_dl: Dataloader containing all query samples for testing.
            max_samples: Maximum number of samples to be used.
            max_batches: Maximum number of batches to be used.
            testdoc: Experiment file
        """
        print("Starting to test the model...\n-------------------------------")
        num_samples = max_samples

        self.generator.eval()
        self.discriminator.eval()

        sequences_per_epoch, target_samples_per_epoch, query_samples_per_epoch, query_labels_per_epoch, out_cf_per_epoch, losses_per_epoch = [], [], [], [], [], []
        batch_count = 0
        num_samples = 0
        with torch.no_grad():
            for data in zip(test_dl, generator_dl):

                X = data[0][0].to(self.device)

                generator_input = data[1][0].to(self.device)
                y_generator_input = data[1][1].to(self.device)

                batch_count += 1
                if batch_count > max_batches:
                    break

                current_batchsize = X.shape[0]
                num_samples += current_batchsize

                feature_mapping = get_feature_mapping(self.args.dataset)

                freeze_indices = [feature_mapping.index(i) for i in self.args.freeze_features]
                remaining_indices = [x for x in range(len(feature_mapping)) if x not in freeze_indices]

                if not feature_mapping:
                    remaining_indices = range(0, 50)

                deltas = self.generator(generator_input)

                deltas_with_freeze = torch.zeros_like(generator_input)
                deltas_with_freeze[:,:,remaining_indices] = deltas

                cf = deltas_with_freeze + generator_input

                out_cf = self.classifier(cf).to(self.device)
                out_cf_class = torch.argmax(out_cf, dim=1)

                cls_loss = compute_classification_loss(out_cf_class, self.args.target_class, device)
                similarity_loss = compute_similarity_loss(deltas_with_freeze)
                sparsity_loss = compute_sparsity_loss(deltas)
                jerk_loss = compute_jerk_loss(deltas_with_freeze, self.device)

                fake_out = self.discriminator(cf)
                adv_loss = self._compute_adversarial_loss(fake_out)
                generator_loss = self._compute_generator_loss(adv_loss, cls_loss, similarity_loss, sparsity_loss, jerk_loss)

                losses = {
                'Classification Loss': cls_loss.detach().cpu().item(),
                'Adversarial Loss': adv_loss.detach().cpu().item(),
                'Similarity Loss': similarity_loss.detach().cpu().item(),
                'Sparsity Loss': sparsity_loss.detach().cpu().item(),
                'Jerk Loss': jerk_loss.detach().cpu().item(),
                'Generator Loss': generator_loss.detach().cpu().item(),
                }

                generated_input = cf
                out_cf_classes = out_cf_class

                sequences_per_epoch.append(generated_input.detach())
                target_samples_per_epoch.append(X.detach())
                query_samples_per_epoch.append(generator_input.detach())
                query_labels_per_epoch.append(y_generator_input.detach())
                out_cf_per_epoch.append(out_cf_classes)
                losses_per_epoch.append(losses)


            avg_generator_loss = get_avg_loss(losses_per_epoch, 'Generator Loss')
            avg_adv_loss = get_avg_loss(losses_per_epoch, 'Adversarial Loss')
            avg_cls_loss = get_avg_loss(losses_per_epoch, 'Classification Loss')
            avg_similarity_loss = get_avg_loss(losses_per_epoch, 'Similarity Loss')
            avg_sparsity_loss = get_avg_loss(losses_per_epoch, 'Sparsity Loss')
            avg_jerk_loss = get_avg_loss(losses_per_epoch, 'Jerk Loss')

            testdoc.append({
            'Dataset': self.args.dataset,
            'Initial Seed': self.args.seed,
            'Number of Epochs': self.args.num_epochs,
            'Target Class': self.args.target_class,
            'Max Batches': self.args.max_batches,
            'Training Batchsize': self.args.batchsize,
            'Learning Rate': self.args.lr,
            'Generator': str(self.generator),
            'Discriminator': str(self.discriminator),
            'Classifier': str(self.classifier),
            'Lambda1': self.args.lambda1,
            'Lambda2': self.args.lambda2,
            'Lambda3': self.args.lambda3,
            'Lambda4': self.args.lambda4,
            'Lambda5': self.args.lambda5,
            'Freeze Features': self.args.freeze_features,
            'Generator Loss': avg_generator_loss,
            'Adversarial Loss': avg_adv_loss,
            'Classification Loss': avg_cls_loss,
            'Similarity Loss': avg_similarity_loss,
            'Sparsity Loss': avg_sparsity_loss,
            'Jerk Loss': avg_jerk_loss
            })

            print(f'[Number of Test Samples: {len(losses_per_epoch)}] [Avg Generator Loss: {avg_generator_loss}] [Avg Adversarial Loss: {avg_adv_loss:.5f}] [Avg Classification Loss: {avg_cls_loss:.5f}] [Avg Similarity Loss: {avg_similarity_loss:.5f}] [Avg Sparsity Loss: {avg_sparsity_loss:.5f}] [Avg Jerk Loss: {avg_jerk_loss:.5f}]  \n')
            
            generated_sequences = []
            for batch in range(len(sequences_per_epoch)):
                for sample in range(len(sequences_per_epoch[batch])):
                    generated_sequences.append(sequences_per_epoch[batch][sample].cpu().numpy())

            original_sequences = []
            for batch in range(len(target_samples_per_epoch)):
                for sample in range(len(target_samples_per_epoch[batch])):
                    original_sequences.append(target_samples_per_epoch[batch][sample].cpu().numpy())

            query_sequences = []
            for batch in range(len(query_samples_per_epoch)):
                for sample in range(len(query_samples_per_epoch[batch])):
                    query_sequences.append(query_samples_per_epoch[batch][sample].cpu().numpy())

            generated_sequences = np.array(generated_sequences)
            original_sequences = np.array(original_sequences)
            query_sequences = np.array(query_sequences)
            now = datetime.now()
            date_and_time = now.strftime("%d_%m_%Y_%H_%M_%S_")
        

            if self.args.save:
                np.save(os.path.join('counterfactuals', self.args.dataset, date_and_time + 'testing_' + self.args.approach + '_counterfactuals.npy'), generated_sequences)
                np.save(os.path.join('counterfactuals', self.args.dataset, date_and_time + 'testing_' + self.args.approach + '_targets.npy'), original_sequences)
                np.save(os.path.join('counterfactuals', self.args.dataset, date_and_time + 'testing_' + self.args.approach + '_queries.npy'), query_sequences)

            return testdoc