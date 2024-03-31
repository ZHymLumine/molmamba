from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import open_dict
import hydra
import torch
import time
import wandb
from transformers import AutoTokenizer

from utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
    get_caduceus_config,
)


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=args.device == "cpu")
    # accelerator = Accelerator(cpu=args.device == "cpu", kwargs_handlers=[ddp_kwargs])
    logger = setup_basics(accelerator, args)
    config = get_caduceus_config(args)

    tokenizer = get_tokenizer(args)
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", model_max_length=512)

    model = get_model(args, config, tokenizer, logger)


    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, shape: {param.shape}")


    param_count = sum(p.numel() for p in model.parameters())
    total_params_millions = param_count / 1e6

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6
    print(f"Before: Total trainable parameters: {total_trainable_params:.2f}M")

    if args.mode == 'ft':
        for name, param in model.named_parameters():
            if 'score' not in name:
                param.requires_grad = False
    
    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6
    print(f"After: Total trainable parameters: {total_trainable_params:.2f}M")

    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)

    
    if args.mode == 'pt':
        train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)
        validation_dataloader = test_dataloader
    elif args.mode == 'ft':
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)


    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    if args.model.compile:
        model = torch.compile(model)

    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer, accelerator)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer, accelerator)
    else:
        train(model, train_dataloader, validation_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)

    logger.finish()


if __name__ == "__main__":
    main()
