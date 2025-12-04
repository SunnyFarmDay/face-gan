from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from progan_modules import Generator, Discriminator


def plot_losses(log_folder, iterations, gen_losses, disc_losses, grad_losses, w_losses, alphas):
    """Plot and save training loss graphs - generates two separate plots"""
    
    # Plot 1: Simple 2x2 layout (original style)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator, Discriminator, and Wasserstein losses
    ax1.plot(iterations, gen_losses, label='Generator', color='blue', alpha=0.7)
    ax1.plot(iterations, disc_losses, label='Discriminator', color='red', alpha=0.7)
    ax1.plot(iterations, w_losses, label='Wasserstein', color='orange', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator vs Discriminator vs Wasserstein Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient penalty
    ax2.plot(iterations, grad_losses, label='Gradient Penalty', color='green', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Penalty')
    ax2.set_title('WGAN-GP Gradient Penalty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Alpha (resolution transition)
    ax3.plot(iterations, alphas, label='Alpha', color='purple', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Alpha')
    ax3.set_title('Resolution Transition (Alpha)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # All metrics combined
    ax4.plot(iterations, gen_losses, label='Generator', color='blue', alpha=0.5, linewidth=2)
    ax4.plot(iterations, disc_losses, label='Discriminator', color='red', alpha=0.5, linewidth=2)
    ax4.plot(iterations, w_losses, label='Wasserstein', color='orange', alpha=0.5, linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(iterations, grad_losses, label='Grad Penalty', color='green', alpha=0.5, linewidth=2, linestyle='--')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('G & D & W Loss')
    ax4_twin.set_ylabel('Gradient Penalty')
    ax4.set_title('All Metrics Combined')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{log_folder}/graphs/training_losses.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Separate Discriminator Loss vs Wasserstein Distance
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator and Discriminator losses (minimization objectives)
    ax5.plot(iterations, gen_losses, label='Generator Loss', color='blue', alpha=0.7)
    ax5.plot(iterations, disc_losses, label='Discriminator Loss', color='red', alpha=0.7)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Loss (Minimization Objective)')
    ax5.set_title('Generator vs Discriminator Loss (What Gets Minimized)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Wasserstein distance (what discriminator maximizes)
    ax6.plot(iterations, w_losses, label='Wasserstein Distance', color='orange', alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Wasserstein Distance')
    ax6.set_title('Wasserstein Distance: E[D(real)] - E[D(fake)]')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Discriminator loss components breakdown
    ax7.plot(iterations, [-w for w in w_losses], label='-Wasserstein', color='orange', alpha=0.7)
    ax7.plot(iterations, grad_losses, label='Gradient Penalty', color='green', alpha=0.7)
    ax7.plot(iterations, disc_losses, label='Total D Loss', color='red', alpha=0.7, linewidth=2)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Loss Components')
    ax7.set_title('Discriminator Loss Breakdown: -W + GP = D_Loss')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Loss balance indicator
    ax8.plot(iterations, gen_losses, label='Generator', color='blue', alpha=0.7)
    ax8.plot(iterations, [-w for w in w_losses], label='-Wasserstein (D objective)', color='orange', alpha=0.7)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Loss')
    ax8.set_title('Training Balance: G Loss vs D Wasserstein Objective')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{log_folder}/graphs/training_losses_detailed.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        # Reduced num_workers to avoid segfaults, added persistent_workers
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=6, persistent_workers=True,
                                 pin_memory=True, prefetch_factor=2)
        return data_loader
    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size+int(image_size*0.2)+1),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader


def train(generator, discriminator, init_step, loader, total_iter=600000, resume_iter=0):
    step = init_step # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)

    #total_iter = 600000
    # When resuming, we still want to train until total_iter, not recalculate
    if resume_iter > 0:
        total_iter_remain = total_iter
    else:
        total_iter_remain = total_iter - (total_iter//6)*(step-1)

    pbar = tqdm(range(resume_iter, total_iter_remain), initial=resume_iter, total=total_iter_remain)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    w_loss_val = 0
    
    # For plotting
    plot_iterations = []
    plot_gen_losses = []
    plot_disc_losses = []
    plot_grad_losses = []
    plot_w_losses = []
    plot_alphas = []

    from datetime import datetime
    import os
    import shutil
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    
    os.mkdir(log_folder)
    os.mkdir(log_folder+'/checkpoint')
    os.mkdir(log_folder+'/sample')
    os.mkdir(log_folder+'/graphs')
    
    # Copy over checkpoint files if resuming
    if resume_iter > 0 and args.checkpoint_g and args.checkpoint_d:
        print(f"Copying checkpoint files to new trial folder...")
        checkpoint_g_dest = f'{log_folder}/checkpoint/{str(resume_iter).zfill(6)}_g.model'
        checkpoint_d_dest = f'{log_folder}/checkpoint/{str(resume_iter).zfill(6)}_d.model'
        shutil.copy2(args.checkpoint_g, checkpoint_g_dest)
        shutil.copy2(args.checkpoint_d, checkpoint_d_dest)
        print(f"Checkpoints copied successfully to {log_folder}/checkpoint/")
        
        # Copy over previous statistics log if exists, filtering to only include data up to resume_iter
        prev_checkpoint_dir = os.path.dirname(args.checkpoint_g)
        prev_trial_dir = os.path.dirname(prev_checkpoint_dir)
        prev_stats_log = os.path.join(prev_trial_dir, 'training_statistics.csv')
        if os.path.exists(prev_stats_log):
            print(f"Copying previous statistics log (filtering to iteration {resume_iter})...")
            import csv
            dest_stats_log = os.path.join(log_folder, 'training_statistics.csv')
            with open(prev_stats_log, 'r') as src_file:
                reader = csv.DictReader(src_file)
                with open(dest_stats_log, 'w', newline='') as dest_file:
                    writer = csv.DictWriter(dest_file, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    for row in reader:
                        if int(row['iteration']) <= resume_iter:
                            writer.writerow(row)
            print(f"Previous statistics log copied successfully (filtered)")

    config_file_name = os.path.join(log_folder, 'train_config_'+post_fix)
    config_file = open(config_file_name, 'w')
    config_file.write(str(args))
    config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_'+post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()
    
    # Statistics log file for detailed analysis
    stats_log_name = os.path.join(log_folder, 'training_statistics.csv')
    if not os.path.exists(stats_log_name):
        stats_log = open(stats_log_name, 'w')
        stats_log.write('iteration,generator_loss,discriminator_loss,wasserstein_loss,gradient_penalty,alpha,step\n')
        stats_log.close()
    
    # Load historical plot data if resuming
    if resume_iter > 0 and os.path.exists(stats_log_name):
        print(f"Loading historical training data for plotting...")
        import csv
        with open(stats_log_name, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iteration = int(row['iteration'])
                # Only load data up to the current resume iteration
                if iteration <= resume_iter:
                    plot_iterations.append(iteration)
                    plot_gen_losses.append(float(row['generator_loss']))
                    plot_disc_losses.append(float(row['discriminator_loss']))
                    plot_w_losses.append(float(row['wasserstein_loss']))
                    plot_grad_losses.append(float(row['gradient_penalty']))
                    plot_alphas.append(float(row['alpha']))
        print(f"Loaded {len(plot_iterations)} historical data points for plotting (up to iteration {resume_iter})")
        
        # Generate initial plot with historical data
        if len(plot_iterations) > 1:
            print(f"Generating initial plot with historical data...")
            plot_losses(log_folder, plot_iterations, plot_gen_losses, 
                       plot_disc_losses, plot_grad_losses, plot_w_losses, plot_alphas)
            print(f"Initial plot generated successfully")

    from shutil import copy
    copy('train.py', log_folder+'/train_%s.py'%post_fix)
    copy('progan_modules.py', log_folder+'/model_%s.py'%post_fix)

    alpha = 0
    #one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    
    # Calculate iteration offset within current step
    iteration = resume_iter % (total_iter // 6) if resume_iter > 0 else 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2/(total_iter//6)) * iteration)

        if iteration > total_iter//6:
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        ### 1. train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(device)
        label = label.to(device)
        real_predict = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, input_code_size).to(device)

        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        
        # Wasserstein distance: E[D(real)] - E[D(fake)]
        w_loss = real_predict - fake_predict
        w_loss_val += w_loss.item()
        
        # Discriminator loss (what's being minimized): -E[D(real)] + E[D(fake)] + GP
        disc_loss = -w_loss.item() + grad_penalty.item()
        disc_loss_val += disc_loss

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()


            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i == resume_iter:
            with torch.no_grad():
                images = g_running(torch.randn(5 * 10, input_code_size).to(device), step=step, alpha=alpha).data.cpu()

                utils.save_image(
                    images,
                    f'{log_folder}/sample/{str(i + 1).zfill(6)}.png',
                    nrow=10,
                    normalize=True,
                    value_range=(-1, 1))
 
        if (i+1) % 10000 == 0 or i == resume_iter:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
            except:
                pass

        if (i+1)%500 == 0:
            g_loss_avg = gen_loss_val/(500//n_critic)
            d_loss_avg = disc_loss_val/500
            grad_loss_avg = grad_loss_val/500
            w_loss_avg = w_loss_val/500
            
            state_msg = (f'{i + 1}; G: {g_loss_avg:.3f}; D: {d_loss_avg:.3f}; W: {w_loss_avg:.3f};'
                f' Grad: {grad_loss_avg:.3f}; Alpha: {alpha:.3f}')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n"%(g_loss_avg, d_loss_avg)
            log_file.write(new_line)
            log_file.close()
            
            # Write detailed statistics to CSV
            stats_log = open(stats_log_name, 'a+')
            stats_line = f"{i + 1},{g_loss_avg:.6f},{d_loss_avg:.6f},{w_loss_avg:.6f},{grad_loss_avg:.6f},{alpha:.6f},{step}\n"
            stats_log.write(stats_line)
            stats_log.close()
            
            # Store data for plotting
            plot_iterations.append(i + 1)
            plot_gen_losses.append(g_loss_avg)
            plot_disc_losses.append(d_loss_avg)
            plot_grad_losses.append(grad_loss_avg)
            plot_w_losses.append(w_loss_avg)
            plot_alphas.append(alpha)
            
            # Update plots every 2000 iterations
            if (i+1) % 2000 == 0 and len(plot_iterations) > 1:
                plot_losses(log_folder, plot_iterations, plot_gen_losses, 
                           plot_disc_losses, plot_grad_losses, plot_w_losses, plot_alphas)

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0
            w_loss_val = 0

            print(state_msg)
            #pbar.set_description(state_msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--path', type=str, help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=300000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    parser.add_argument('--checkpoint_g', type=str, default=None, help='path to generator checkpoint to resume from')
    parser.add_argument('--checkpoint_d', type=str, default=None, help='path to discriminator checkpoint to resume from')
    parser.add_argument('--resume_iter', type=int, default=0, help='iteration number to resume from')
    
    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    device = torch.device("cuda:%d"%(args.gpu_id))
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic

    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    
    ## Load checkpoints if provided
    if args.checkpoint_g and args.checkpoint_d:
        print(f"Loading generator checkpoint: {args.checkpoint_g}")
        print(f"Loading discriminator checkpoint: {args.checkpoint_d}")
        generator.load_state_dict(torch.load(args.checkpoint_g))
        g_running.load_state_dict(torch.load(args.checkpoint_g))
        discriminator.load_state_dict(torch.load(args.checkpoint_d))
        print("Checkpoints loaded successfully!")
    
    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    loader = imagefolder_loader(args.path)

    train(generator, discriminator, args.init_step, loader, args.total_iter, args.resume_iter)
