from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client

# new imports for defenses
from federated_learning import aggregators
from copy import deepcopy
import torch


def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    # train selected clients
    for client_idx in random_workers:
        args.get_logger().info(
            "Training epoch #{} on client #{}",
            str(epoch),
            str(clients[client_idx].get_client_index()),
        )
        clients[client_idx].train(epoch)

    args.get_logger().info("Aggregating client parameters")

    # collect parameters from selected clients
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]

    # --- Defense-aware aggregation ---
    global_params = deepcopy(clients[0].get_nn_parameters())  # server’s reference
    deltas = []
    for client_params in parameters:
        delta = {}
        for k in client_params.keys():
            delta[k] = (
                client_params[k].detach().cpu()
                - global_params[k].detach().cpu()
            )
        deltas.append(delta)

    agg_name = getattr(args, "aggregator", "simple")
    if agg_name == "simple":
        # default simple average
        new_nn_params = average_nn_parameters(parameters)
        info = {"filtered_count": 0, "kept_indices": list(range(len(parameters)))}
    elif agg_name == "norm_filter":
        agg_delta, info = aggregators.norm_filter_then_average(
            deltas, z_threshold=getattr(args, "z_threshold", 2.0)
        )
        new_nn_params = {}
        for k in global_params.keys():
            device = global_params[k].device
            new_nn_params[k] = (
                global_params[k].detach().cpu() + agg_delta[k].to(device)
            ).to(device)
    elif agg_name == "some_mean":
        agg_delta, info = aggregators.some_mean_filter_then_average(
            deltas, z_threshold=getattr(args, "z_threshold", 2.0)
        )
        new_nn_params = {}
        for k in global_params.keys():
            device = global_params[k].device
            new_nn_params[k] = (
                global_params[k].detach().cpu() + agg_delta[k].to(device)
            ).to(device)
    elif agg_name == "median":
        agg_delta, info = aggregators.median_filter_then_average(deltas)
        new_nn_params = {}
        for k in global_params.keys():
            device = global_params[k].device
            new_nn_params[k] = (
                global_params[k].detach().cpu() + agg_delta[k].to(device)
            ).to(device)
    
    elif agg_name == "krum":
        agg_delta, info = aggregators.krum_filter_then_average(
            deltas, f=args.get_num_poisoned_workers(), m=1
        )
        new_nn_params = {}
        for k in global_params.keys():
            device = global_params[k].device
            new_nn_params[k] = (
                global_params[k].detach().cpu() + agg_delta[k].to(device)
            ).to(device)

    elif agg_name == "trimmed_mean":
        # b is typically the number of faulty clients, same as f
        b = args.get_num_poisoned_workers()
        agg_delta, info = aggregators.trimmed_mean_filter_then_average(deltas, b=b)
        new_nn_params = {}
        for k in global_params.keys():
            device = global_params[k].device
            new_nn_params[k] = (
                global_params[k].detach().cpu() + agg_delta[k].to(device)
            ).to(device)
    else:
        new_nn_params = average_nn_parameters(parameters)
        info = {"filtered_count": 0, "kept_indices": list(range(len(parameters)))}

    # update all clients with new global params
    for client in clients:
        args.get_logger().info(
            "Updating parameters on client #{}", str(client.get_client_index())
        )
        client.update_nn_parameters(new_nn_params)

    # log defense info
    try:
        filtered = info.get("filtered_count", 0)
        kept = info.get("kept_indices", None)
        args.get_logger().info(
            "[Aggregation] method={} filtered={} kept={}",
            agg_name,
            filtered,
            kept,
        )
    except Exception:
        pass

    return clients[0].test(), random_workers


def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))
    return clients


def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(
            epoch, args, clients, poisoned_workers
        )
        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results), worker_selection


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    if 'aggregator' in KWARGS:
        setattr(args, 'aggregator', KWARGS['aggregator'])
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(
        args.get_num_workers(), args.get_num_poisoned_workers()
    )
    distributed_train_dataset = poison_data(
        logger, distributed_train_dataset, args.get_num_workers(),
        poisoned_workers, replacement_method
    )

    train_data_loaders = generate_data_loaders_from_distributed_dataset(
        distributed_train_dataset, args.get_batch_size()
    )

    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
