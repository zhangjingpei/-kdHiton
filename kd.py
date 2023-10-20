import torch
import torch.nn.functional
from torch import nn

from labml import experiment, tracker
from labml.configs import option
from labml_helpers.train_valid import BatchIndex
from labml_nn.distillation.large import LargeModel
from labml_nn.distillation.small import SmallModel
from labml_nn.experiments.cifar10 import CIFAR10Configs


class Configs(CIFAR10Configs):
    '''
    Configs 是一个类，用于配置深度学习实验的参数。
    CIFAR10Configs，这表明它包括了有关CIFAR-10数据集的配置信息。
    '''

    """
    ## Configurations

    This extends from [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """
    # The small model
    model: SmallModel
    # The large model
    large: LargeModel
    # KL Divergence loss for soft targets
    kl_div_loss = nn.KLDivLoss(log_target=True)
    # Cross entropy loss for true label loss
    loss_func = nn.CrossEntropyLoss()
    # Temperature, $T$
    temperature: float = 5.
    # Weight for soft targets loss.
    #
    # The gradients produced by soft targets get scaled by $\frac{1}{T^2}$.
    # To compensate for this the paper suggests scaling the soft targets loss
    # by a factor of $T^2$
    soft_targets_weight: float = 100.
    # Weight for true label cross entropy loss
    label_loss_weight: float = 0.5

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training/validation step

        We define a custom training/validation step to include the distillation
        """

        # Training/Evaluation mode for the small model
        self.model.train(self.mode.is_train)
        # Large model in evaluation mode
        self.large.eval()

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of samples processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Get the output logits, $v_i$, from the large model
        with torch.no_grad():
            large_logits = self.large(data)

        # Get the output logits, $z_i$, from the small model
        output = self.model(data)

        # Soft targets
        # $$p_i = \frac{\exp (\frac{v_i}{T})}{\sum_j \exp (\frac{v_j}{T})}$$
        soft_targets = nn.functional.log_softmax(large_logits / self.temperature, dim=-1)
        # Temperature adjusted probabilities of the small model
        # $$q_i = \frac{\exp (\frac{z_i}{T})}{\sum_j \exp (\frac{z_j}{T})}$$
        soft_prob = nn.functional.log_softmax(output / self.temperature, dim=-1)

        # Calculate the soft targets loss
        soft_targets_loss = self.kl_div_loss(soft_prob, soft_targets)
        # Calculate the true label loss
        label_loss = self.loss_func(output, target)
        # Weighted sum of the two losses
        loss = self.soft_targets_weight * soft_targets_loss + self.label_loss_weight * label_loss
        # Log the losses
        tracker.add({"loss.kl_div.": soft_targets_loss,
                     "loss.nll": label_loss,
                     "loss.": loss})

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(Configs.large)
def _large_model(c: Configs):
    """
    ### Create large model
    """
    return LargeModel().to(c.device)


@option(Configs.model)
def _small_student_model(c: Configs):
    """
    ### Create small model
    """
    return SmallModel().to(c.device)


def get_saved_model(run_uuid: str, checkpoint: int):
    """
    ### Load [trained large model](large.html)
    """

    from labml_nn.distillation.large import Configs as LargeConfigs

    # In evaluation mode (no recording)
    experiment.evaluate()
    # Initialize configs of the large model training experiment
    conf = LargeConfigs()
    # Load saved configs
    experiment.configs(conf, experiment.load_configs(run_uuid))
    # Set models for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Set which run and checkpoint to load
    experiment.load(run_uuid, checkpoint)
    # Start the experiment - this will load the model, and prepare everything
    experiment.start()

    # Return the model
    return conf.model


def main(run_uuid: str, checkpoint: int):
    """
    Train a small model with distillation
    """
    # Load saved model
    large_model = get_saved_model(run_uuid, checkpoint)
    # Create experiment
    experiment.create(name='distillation', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Set the loaded large model
    conf.large = large_model
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'model': '_small_student_model',
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Start experiment from scratch
    experiment.load(None, None)
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main('d46cd53edaec11eb93c38d6538aee7d6', 1_000_000)
