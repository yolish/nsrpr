"""
Entry point training and testing NS-RPR
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.KNNCameraPoseDataset import KNNCameraPoseDataset
from models.NSRPR import NSRPR
from models.pose_losses import CameraPoseLoss
from os.path import join
import transforms3d as t3d


def compute_abs_pose(rel_pose, abs_pose_neighbor, device):
    # p_neighbor p_rel = p_query
    # p1 p_rel = p2
    abs_pose_query = torch.zeros_like(rel_pose)
    rel_pose = rel_pose.cpu().numpy()
    abs_pose_neighbor = abs_pose_neighbor.cpu().numpy()
    for i, rpr in enumerate(rel_pose):
        p1 = abs_pose_neighbor[i]

        t_rel = rpr[:3]
        q_rel = rpr[3:]
        rot_rel = t3d.quaternions.quat2mat(q_rel/ np.linalg.norm(q_rel))

        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1/ np.linalg.norm(q1))

        t2 = t1 + t_rel
        rot2 = np.dot(rot1,rot_rel)
        q2 = t3d.quaternions.mat2quat(rot2)
        abs_pose_query[i][:3] = torch.Tensor(t2).to(device)
        abs_pose_query[i][3:] = torch.Tensor(q2).to(device)

    return abs_pose_query


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("query_labels_file", help="path to a query file mapping images to their poses, scenes, netvlad etc")
    arg_parser.add_argument("db_labels_file", help="path to a db file mapping images to their poses, scenes, netvlad etc")
    arg_parser.add_argument("knn_file",
                            help="path to a file with the dn knn for each query")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained NS-RPR model")
    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} experiment for NS-RPR".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using QUERY labels file: {}".format(args.query_labels_file))
    logging.info("Using DB labels file: {}".format(args.db_labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the RPR model
    model = NSRPR(config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        train_dataset = KNNCameraPoseDataset(args.dataset_path, args.query_labels_file, args.db_labels_file,
                                             args.knn_file, sample_neighbors=True, sample_size=config.get("k"))

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        # Resetting temporal loss used for logging
        running_loss = 0.0
        n_samples = 0

        for epoch in range(n_epochs):
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                query = minibatch['query'].to(dtype=torch.float32)
                knn = minibatch['knn'].to(dtype=torch.float32)
                gt_rel_poses = minibatch['rel_poses'].to(dtype=torch.float32)
                batch_size = gt_rel_poses.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Estimate the relative pose
                # Zero the gradients
                optim.zero_grad()

                res = model(query, knn)
                knn_distr_before = res["knn_distr_before"]
                knn_distr_after = res["knn_distr_after"]

                best_neighbors = torch.zeros((batch_size, 1)).to(device).to(dtype=torch.long)
                # Compute the total loss weighted by the after distribution
                for query_idx in range(batch_size):
                    min_pose_criterion_index = -1
                    min_pose_criterion = 0
                    my_knn_w = torch.exp(knn_distr_after[query_idx, :])
                    my_pose_criterion = 0
                    for i in range(knn.shape[1]):
                        loss_i = pose_loss(res["rel_pose_{}".format(i)][query_idx].unsqueeze(1),
                                           gt_rel_poses[query_idx, i, :].unsqueeze(1))
                        if min_pose_criterion_index == -1:
                            #best_w = my_knn_w[i]
                            min_pose_criterion_index = 0
                            min_pose_criterion = loss_i.item()
                            my_pose_criterion = my_knn_w[i] * loss_i
                        else:
                            #if my_knn_w[i] >= best_w:
                            #    my_pose_criterion = loss_i
                            my_pose_criterion = my_pose_criterion + my_knn_w[i]*loss_i
                            if loss_i <= min_pose_criterion:
                                min_pose_criterion = loss_i.item()
                                min_pose_criterion_index = i
                    best_neighbors[query_idx] = min_pose_criterion_index

                    if query_idx == 0:
                        total_pose_criterion = my_pose_criterion
                        posit_err, orient_err = utils.pose_err(
                            res["rel_pose_{}".format(min_pose_criterion_index)][query_idx].detach().unsqueeze(0),
                            gt_rel_poses[query_idx,min_pose_criterion_index].detach().unsqueeze(0))
                        posit_err = posit_err.item()
                        orient_err = orient_err.item()

                    else:
                        total_pose_criterion = total_pose_criterion + my_pose_criterion
                        p_err, o_err = utils.pose_err(
                            res["rel_pose_{}".format(min_pose_criterion_index)][query_idx].detach().unsqueeze(0),
                            gt_rel_poses[query_idx,min_pose_criterion_index].detach().unsqueeze(0))
                        posit_err = posit_err + p_err.item()
                        orient_err = orient_err + o_err.item()

                total_pose_criterion = total_pose_criterion / batch_size
                posit_err = posit_err / batch_size
                orient_err = orient_err / batch_size
                # Compute the NLL loss

                criterion = total_pose_criterion + nll_loss(knn_distr_before, best_neighbors) + \
                            nll_loss(knn_distr_after, best_neighbors)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:

                    msg = "[Batch-{}/Epoch-{}] running relative camera pose loss: {:.3f}, camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_freq_print),
                                                                        posit_err,
                                                                        orient_err)

                    logging.info(msg)
                    # Resetting temporal loss used for logging
                    running_loss = 0.0
                    n_samples = 0

            # Save checkpoint3n
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_nsrpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_nsrpr_final.pth')

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        test_dataset = KNNCameraPoseDataset(args.dataset_path, args.query_labels_file, args.db_labels_file,
                                             args.knn_file, sample_neighbors=False, sample_size=config.get("k"))
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
        abs_stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                query = minibatch['query'].to(dtype=torch.float32)
                knn = minibatch['knn'].to(dtype=torch.float32)
                gt_query_pose = minibatch['query_pose'].to(dtype=torch.float32)
                knn_poses = minibatch['knn_poses'].to(dtype=torch.float32)

                # Select the best neighbor and take the estimated relative pose
                tic = time.time()
                res = model(query, knn)
                knn_distr_after = res["knn_distr_after"]
                selected_nbr_index = torch.argmax(knn_distr_after).item()
                est_rel_pose = res["rel_pose_{}".format(selected_nbr_index)]

                est_pose = compute_abs_pose(est_rel_pose, knn_poses[0, selected_nbr_index].unsqueeze(0), device)


                torch.cuda.synchronize()
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_query_pose)

                # Collect statistics
                abs_stats[i, 0] = posit_err.item()
                abs_stats[i, 1] = orient_err.item()
                abs_stats[i, 2] = (toc - tic)*1000

                logging.info("Absolute Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    abs_stats[i, 0],  abs_stats[i, 1],  abs_stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.query_labels_file))
        logging.info("Median absolute pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(abs_stats[:, 0]),
                                                                                 np.nanmedian(abs_stats[:, 1])))
        logging.info("Mean pose inference time:{:.2f}[ms]".format(np.mean(abs_stats[:, 2])))






