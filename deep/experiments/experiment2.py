import os.path
import torch
import torchvision.models
from torch import nn
from data import make_dataloader
from train import train, evaluate
import sys
import shutil
import utils
from utils import State, Reporter, Table
from backtest.data import JSONCache, FetchCharts, Cache, FilterNoneCharts, CausalImpute
from pipes import Finviz, RmTz
from torchinfo import summary
from datetime import datetime
from metrics import custom_precision, accuracy, precision_2d
import models
# To verify if the config has the good format
from configs.formats import config_format


metrics = {
    "accuracy": accuracy,
    "precision": precision_2d
}

def experiment2(args, kwargs):
    if args.dataset == "huge":
        pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=ipodate_more5",True) | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
    elif args.dataset == "small":
        pipe = Finviz("https://finviz.com/screener.ashx?v=111&f=cap_mega",True) | JSONCache() | \
               FetchCharts(progress=True, throttle=1., auto_adjust=False) | \
               Cache() | FilterNoneCharts() | RmTz() | CausalImpute()
        pipe.set_id(10)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # Setup
    device = utils.get_device() if not args.cpu else "cpu"
    hyper = utils.clean_dict(vars(args).copy())
    hyper.update(kwargs)

    DEBUG = args.debug

    # Loading the config file
    config = utils.ConfigFile(args.config, config_format, verify_path=True)
    config.override_config(kwargs)

    # Preparing Result Table
    rtable = Table("results/resultTable.json")
    sys.excepthook = rtable.handle_exception(sys.excepthook)

    if DEBUG:
        run_id = "DEBUG"
    else:
        resultSocket = rtable.registerRecord(__name__, args.config, category=None, **hyper)
        run_id = resultSocket.get_run_id()

    config["model"]["model_dir"] = f'{config["model"]["model_dir"]}/{run_id}'

    comment = '' if hyper.get("comment") is None else hyper.get("comment")
    if os.path.exists(f'runs/{run_id}'):
        print(f"Clearing tensorboard logs for id: {run_id}")
        shutil.rmtree(f'runs/{run_id}')
    State.writer = Reporter(log_dir=f'runs/{run_id}', comment=comment)

    # Loading the data
    train_loader, val_loader, test_loader = make_dataloader(config=config, pipe=pipe,
                                                            start=datetime(2000, 1, 1),
                                                            train_end=datetime(2016, 12, 31),
                                                            val_end=datetime(2019, 12, 13),
                                                            test_end=datetime(2024, 1, 1),
                                                            fract=args.fract, annotation_type="change")
    print("Data loaded successfully!")

    # Loading the model
    model = models.from_name(config, annotation_type="change")
    model = model.to(device)
    summary(model, input_size=(config["data"]["batch_size"], 3, 2 * config["data"]["p_quant"], 2 * config["data"]["window_len"]), device=device)
    print("Model loaded successfully!")

    # Loading optimizer, loss and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                 weight_decay=config["training"]["weight_decay"])
    loss = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=config["training"]["num_epochs"],
                                                           eta_min=config["training"]["min_lr"])
    # Training
    print("Begining training...")
    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss,
        num_epochs=config["training"]["num_epochs"],
        device=device,
        scheduler=scheduler,
        config=config,
        metrics=metrics
    )
    # Test
    results = evaluate(model, test_loader, loss, device, metrics=metrics)
    print("Training done!  Saving...")

    save_dict = {
        "epoch": config["training"]["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_results": results
    }
    torch.save(
        save_dict, f"{config['model']['model_dir']}/final_{config['model']['name']}.pth"
    )

    # Copy config file to model dir
    shutil.copy(args.config, config['model']["model_dir"])

    # Print stats of code
    print(config.stats())
    print(f"{utils.Color(11)}{State.warning()}{utils.ResetColor()}")

    State.writer.flush()
    State.writer.close()

    # Save results
    if not DEBUG:
        resultSocket.write(accuracy=results["accuracy"], crossEntropy=results["loss"], precision=results["precision"])
        rtable.toTxt()
