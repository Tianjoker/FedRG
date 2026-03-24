import torch
from loss.CE_loss import CrossEntropyLoss
from loss.SymCE_loss import SCELoss

from omegaconf import OmegaConf
import logging

class LossManager:
    def __init__(self, args, device,**kwargs):
        self.args = args
        self.device = device
        self.loss_functions = {}
        self.weights = {}
        self.loss_params = {}
        self.extra_kwargs = kwargs

        if not hasattr(args.algorithms, 'loss_functions') or not self.args.algorithms.loss_functions:
            logging.warning("Using default cross_entropy loss")
            self.loss_functions['cross_entropy'] = CrossEntropyLoss()
            self.weights['cross_entropy'] = 1.0
            self.loss_params['cross_entropy'] = ['outputs', 'labels']
        else:
            for loss_name, weight in self.args.algorithms.loss_functions.items():
                if loss_name == 'cross_entropy':
                    self.loss_functions[loss_name] = CrossEntropyLoss()
                    self.loss_params[loss_name] = ['outputs', 'labels']
                elif loss_name == 'symmetricce':
                    symmetricce_config = OmegaConf.load("conf/algorithms/symmetricCE.yaml")
                    if not hasattr(symmetricce_config, 'alpha') or not hasattr(symmetricce_config, 'beta') or not hasattr(self.args.datasets, 'num_classes'):
                        logging.error("Missing alpha, beta, or num_classes in symmetricCe.yaml, skipping")
                        continue
                    self.loss_functions[loss_name] = SCELoss(
                        alpha=symmetricce_config.alpha,
                        beta=symmetricce_config.beta,
                        device=self.device,
                        num_classes=self.args.datasets.num_classes
                    )
                    self.loss_params[loss_name] = ['outputs', 'labels']
                else:
                    logging.warning(f"Unknown loss function: {loss_name}, skipping")
                    continue
                self.weights[loss_name] = weight
                logging.info(f"Initialized loss function: {loss_name} with params: {self.loss_params[loss_name]}")

    def compute_loss(self, **kwargs):
        if kwargs.get('twin_model', False):
            total_loss = (
                torch.tensor(0.0, device=self.device, requires_grad=True),
                torch.tensor(0.0, device=self.device, requires_grad=True)
            )
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        individual_losses = {}

        for loss_name, loss_fn in self.loss_functions.items():

            required_param_combos = self.loss_params.get(loss_name, [])

            if isinstance(required_param_combos[0], str):
                required_param_combos = [required_param_combos]


            matched_params = None
            for param_set in required_param_combos:
                if all(p in kwargs for p in param_set):
                    matched_params = param_set
                    break

            if matched_params is None:
                logging.warning(f"No matching parameter set found for {loss_name}, skipping")
                continue

            fn_kwargs = {param: kwargs[param] for param in matched_params}


            result = loss_fn(**fn_kwargs)


            if torch.is_tensor(result) and result.ndim > 0:

                logging.debug(f"{loss_name} returned batch loss shape {result.shape}, taking mean")
                result = result.mean()

            weight = self.weights.get(loss_name, 1.0)

            if isinstance(result, tuple):
                tensor_losses = [x for x in result if isinstance(x, torch.Tensor) and x.requires_grad]
                if not tensor_losses:
                    logging.warning(f"{loss_name} returned no valid Tensor losses, skipping")
                    continue

                # Store sub-losses
                individual_losses[loss_name] = result
                logging.debug(f"{loss_name} loss: {result}")
                # Update total_loss
                total_loss = (
                    total_loss[0] + weight * tensor_losses[0],
                    total_loss[1] + weight * tensor_losses[1]
                )
            else:
                if not isinstance(result, torch.Tensor):
                    logging.warning(f"{loss_name} returned a non-Tensor value: {result}, converting to Tensor")
                    result = torch.tensor(float(result), device=self.device, requires_grad=True)
                individual_losses[loss_name] = result
                logging.debug(f"{loss_name} loss: {result.item()}")
                if isinstance(total_loss, tuple):
                    result_1 = result.detach().clone().requires_grad_(True)
                    result_2 = result.detach().clone().requires_grad_(True)
                    total_loss = (
                        total_loss[0] + weight * result_1,
                        total_loss[1] + weight * result_2
                    )
                else:
                    total_loss = total_loss + weight * result
        return total_loss, individual_losses