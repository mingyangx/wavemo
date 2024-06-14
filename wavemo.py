from network.UNetZern import ZernUNet
from network.UNetFredo import FredoUNet
from network.UNetVGG16 import VGG16Unet
from network.Restormer import Restormer
from network.UNetAttMulti import AttU_Net_Multi
from network.DnCNN import DnCNN
from data import *
from piq import ssim, psnr
from operator import itemgetter 
from torchvision import transforms
from torchmetrics.image import TotalVariation
from modulate import *  
from torchvision.utils import save_image
import scipy.io as sio
import time
from copy import copy
from utils import *
import gc
from network.SirenMulti import MultiSiren
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm


def wavemo(
        use_modulation=True, 
        sim=True,
        overwrite_ckpts=False,
        learn_slm=True, 
        random_abe=False, 
        slm_mode='mlp', # DIP
        slm_loss_weight=0, #0.27
        tv_weight=0,
        save_folder='Test', 
        save_root='output',
        resume_ckpt_path=None, 
        num_epochs=1, # currently one epoch takes around 12 hours on MIT Places data_large, which appears to be enough.
        train_size=1e7, # 1e5 per gpu hour
        nframe = 32,   
        accum_batch = 1, 
        batch_size = 8,
        residual = True, 
        zern_std_tuple = (5, 6),  
        rand_std_tuple = (2., 2.5),
        net_arch = 'attention', # Fredo / Attention / Restormer
        grid_size = 1,
        dataset = 'places', # places / fashion,
        zern_order = 7,
        img_scale = 1,  # don't use this, use use_low_res_fashion instead
        real_data_subfolder = '',
        use_low_res_fashion = False,
        training_data_dir = '',
        eval_only = False,
        mtf_loss = True,
        mtf_weight = 0.,
        mtf_mode = 'one', # or ratio
        test_with_tissue = False,
        zern_insertion = 'none',
        rand_zero_init = True,
        mlp_hidden_layers = 1,
        mlp_act = 'leakyrelu',
        captured_data_dir = None,
        hidden_dim = 32,
        low_pass_img = False,
        low_pass_level = 25,
        zern_basis_scalar = 1,
        mlp_init_scalar = 1,
        focus_std=0.2,
        only_resume_slm=False,
        use_wandb = False,
    
        ):
    

    Path(save_root).mkdir(parents=True, exist_ok=True)
    seed_torch(0)
    img_size = 256 // img_scale  # only for simulated data

    if 'fashion' in dataset.lower() and use_low_res_fashion:
        img_size = 32
        img_scale = 1

    if slm_mode.lower() == 'siren':
        batch_size = 1

    if batch_size == 1:
        accum_batch = 4

    if not sim:
        learn_slm = False

    if sim:
        if 'places' in dataset.lower():
            raw_size_used = 100 // img_scale  
            padded_size = int(128) // img_scale  

        elif 'fashion' in dataset.lower(): 
            raw_size_used = 28 // img_scale  
            padded_size = int(32) // img_scale  
        else:
            print('Unknown dataset!, please choose from places or fashion!')
            raise NotImplementedError
        
        equiv_center_size = int(raw_size_used / padded_size * img_size) 
        crop_ROI = transforms.CenterCrop(int(equiv_center_size))

    if sim:
        save_per_minutes = 60
        test_per_minutes = 10
    else:
        save_per_minutes = 30
        test_per_minutes = 10
    abe_std_low, abe_std_high = zern_std_tuple[0], zern_std_tuple[1]
    rand_std_low, rand_std_high = rand_std_tuple[0], rand_std_tuple[1]  
    slm_std = 0.5*(abe_std_low + abe_std_high)
    mask_std = 32 
    mask_size = round_up_to_odd(img_size//2)    
    save_frequency = 1   
    save_checkpoint_freq = 1       
    
    if sim:
        test_size = 1000 # originally 1000      # The first n elements  
    else:
        test_size = 20

    # other parameters!
    print('PyTorch Version:', torch.__version__)
    root_dir = f'{save_root}' # root folder, the folder for this experiment will be created inside this root folder
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    
    if not sim:
        real_data_dir = f'{captured_data_dir}/train'
        new_abe_dir = f'{captured_data_dir}/test_same_slide'
        new_slide_dir = f'{captured_data_dir}/test_diff_slide'
        crop_ROI = transforms.CenterCrop(int(400))

    tissue_dir = None
    sim_dir = training_data_dir

    if not use_modulation:
        learn_slm = False

    g_lr_milestones = [1e10]
    slm_lr_milestones = [1e10]
    verbose = False 
    train_size = int(train_size)

    if resume_ckpt_path is None or resume_ckpt_path == '':
        resume = False
        resume_name = ''
    else:
        resume = True
        resume_name = '_Resume'

    if resume:
        init_lr=1e-4
    else:
        init_lr=1e-3
            
    if sim:
        init_lr = 1e-3

    final_lr=1e-6

    if sim:
        num_epochs = 1
        if 'fashion' in dataset:
            num_epochs = 50
    else:
        num_epochs = 100

    disable_inner_tqdm = False if sim else True
    sim_name = 'Sim' if sim else 'Real'
    train_name = 'LearnSLM' if learn_slm else 'UnOptSLM'
    if random_abe:
        random_abe_name = f'RandAbe_{rand_std_low}_{rand_std_high}'
    else:
        random_abe_name = f'ZernAbe_{abe_std_low}_{abe_std_high}'

    mtf_name = f'MTF_{mtf_mode}' if mtf_loss else 'NoMTF'
    slm_name = f'SLM_{slm_mode}_{mtf_name}_{mtf_loss}_{mtf_weight}'
    if slm_mode.lower() == 'mlp':
        slm_name = f'SLM_mlp_{mlp_hidden_layers}_{hidden_dim}'
    if slm_mode.lower() == 'dip':
        slm_name = f'SLM_dip_zern_{zern_insertion}_{mtf_name}_{mtf_loss}_{mtf_weight}'
    if slm_mode.lower() == 'random' and rand_zero_init:
        slm_name = f'SLM_rand_zero_init_{mtf_name}_{mtf_loss}_{mtf_weight}'

    exp_name = f'{train_name}' if sim else 'Exp'
    use_modulation_name = f'{exp_name}' if use_modulation else 'NoMod'
    real_mode_name = f'_{real_data_subfolder}' if real_data_subfolder != '' else ''
    test_data_name = 'Tissue' if test_with_tissue else 'Test'
    low_pass_name = f'_LowPass_{low_pass_level}_' if low_pass_img else ''
    if sim:
        sim_name += '_' + test_data_name
    
    if not use_modulation:
        nframe = 0
    
    save_folder = f'{save_folder}_{sim_name}{real_mode_name}_{resume_name}_{use_modulation_name}_{slm_name}_{random_abe_name}_N{nframe}'
    assert slm_mode.lower() in ['zern', 'random', 'mlp', 'dip', 'siren', 'focus'], 'slm_mode must be zern, random or mlp!'

    #---------------------------------------------------------------------------------------
    #--- environment 
    #---------------------------------------------------------------------------------------
    seed_torch(0)
    init_env()
    device = torch.device('cuda') # device for training
    save_dir = os.path.join(root_dir, save_folder)
    num_workers = 0 # when using debugger, num_workers must be 0

    if use_wandb:
        try:
            run = wandb.init(project="WaveMo", name=save_folder, reinit=True, dir=f'{root_dir}/wandb', settings=wandb.Settings(start_method="fork"))
        except:
            use_wandb = False
        
    # print(colored(f'use_wandb: {use_wandb}', 'red'))
    create_save_folder(save_dir, verbose=True)
    save_ckpt_dir = f'{save_dir}/checkpoints' 
    Path(save_ckpt_dir).mkdir(parents=True, exist_ok=True)
    for temp_name in ['epochs']:   # 'checkpoints', 
        Path(f'{save_dir}/{temp_name}').mkdir(parents=True, exist_ok=True)
    Path(f'{save_ckpt_dir}/mat').mkdir(parents=True, exist_ok=True)
    Path(f'{save_ckpt_dir}/pth').mkdir(parents=True, exist_ok=True)
    save_checkpoint_freq = save_checkpoint_freq // save_frequency * save_frequency
    if not use_modulation:
        save_checkpoint_freq *= 2
        save_checkpoint_freq = int(save_checkpoint_freq)

    #-------------------------------------------------------------------------------------
    #--- prepare for training
    #-------------------------------------------------------------------------------------
    net_channels = nframe + 1 if use_modulation else 1

    if 'fredo' in net_arch.lower():
        model = FredoUNet(in_channel=1,out_channel=1, residual=True, nframes=net_channels, permutation_invariant=False)
    elif 'attention' in net_arch.lower():
        model = AttU_Net_Multi(img_ch=net_channels, output_ch=1, residual=residual)
    elif 'vanilla' in net_arch.lower():
        model = VGG16Unet(input_channel=net_channels, vgg_weights=None, residual=residual)
    elif 'restormer' in net_arch.lower():
        model = Restormer(img_ch=net_channels, output_ch=1, residual=residual,
                          num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[32, 64, 128, 256], 
                          num_refinement=4, expansion_factor=2)                   
    elif 'dncnn' in net_arch.lower():
        model = DnCNN(img_ch=net_channels, output_ch=1, residual=residual, n_layers=11) 
    else:
        print('Unknown network architecture!')
        print('Please choose from Fredo / Attention / Restormer!')
        raise NotImplementedError

    print('Using network architecture:', net_arch)
    model = model.to(device)

    if not sim:
        use_modulation = False

    if 'debug' in save_folder.lower() and sim:
        save_per_minutes = 1
        test_per_minutes = 1
        sim_dir = sim_dir.replace('data_large', 'val_large')
        # train_size = max(64, int(batch_size * 2))
        test_size = max(32, batch_size)

    if resume and not only_resume_slm:
        model.load_state_dict(torch.load(resume_ckpt_path)['state_dict'])
        print(colored(f'Loaded previous checkpoint from path: {resume_ckpt_path}', 'red'))

    criterion = nn.MSELoss()
    save_model_name = None
    tv = TotalVariation().to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    if 'restormer' in net_arch.lower():
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e5, eta_min=final_lr)

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, g_lr_milestones, gamma=0.1)

    #-------------------------------------------------------------------------------------
    #--- load data
    #-------------------------------------------------------------------------------------
    print('Loading data ...')
    load_start_time = time.time()
    if not sim:
        transform = None
        places_dataset = FFN(data_dir=real_data_dir, input_transforms=transform, nframe=nframe, use_transforms=True, train=True, use_modulation=use_modulation, child_folder=real_data_subfolder) 
    if sim:
        transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(int(raw_size_used)), transforms.CenterCrop(int(padded_size)), transforms.Resize(size=(img_size, img_size))])
        if dataset.lower() == 'places':
            places_dataset = FFN_Sim(data_folder=sim_dir, input_transforms=transform)
        else:
            print('Unknown dataset!, please choose from places or fashion!')
            raise NotImplementedError
        tissue_dataset = FFN_Sim(data_folder=tissue_dir, input_transforms=transform)

    load_finish_time = time.time()
    print(f'Finished loading data in {(load_finish_time - load_start_time)/60:.2f} minutes')
    img_inds = np.arange(len(places_dataset))
    seed_torch(0)
    np.random.shuffle(img_inds)

    train_inds = img_inds[:int(train_size)]
    test_inds = img_inds[-int(test_size):]
    seed_torch(22)
    np.random.shuffle(train_inds)
    if not sim:
        print('Test GT Filenames', itemgetter(*test_inds)(places_dataset.data))
    train_dataset = torch.utils.data.Subset(places_dataset, train_inds)
    test_dataset = torch.utils.data.Subset(places_dataset, test_inds)
    if sim:
        tissue_dataset = torch.utils.data.Subset(tissue_dataset, np.arange(test_size))
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    
    if not test_with_tissue:
        test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, drop_last=True) 
    else:
        test_loader = DataLoader(tissue_dataset, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    print('train size', len(train_dataset))

    if not sim:
        transform = None

        new_abe_dataset = FFN(data_dir=new_abe_dir  , input_transforms=transform, nframe=nframe, use_transforms=True, train=False, use_modulation=use_modulation, child_folder=real_data_subfolder) 
        new_abe_loader = DataLoader(new_abe_dataset, num_workers=num_workers, batch_size=1, drop_last=True)

        new_slide_dataset = FFN(data_dir=new_slide_dir  , input_transforms=transform, nframe=nframe, use_transforms=True, train=False, use_modulation=use_modulation, child_folder=real_data_subfolder) 
        new_slide_loader = DataLoader(new_slide_dataset, num_workers=num_workers, batch_size=1, drop_last=True)

    #-------------------------------------------------------------------------------------
    #--- generate aberration function
    #-------------------------------------------------------------------------------------
    zernike_basis = generate_zernike_basis(width=img_size, zern_order=zern_order).to(device)

    if sim and use_modulation:

        if slm_mode.lower() == 'zern':
            slm_alphas = (slm_std * torch.rand(nframe, (zern_order*(zern_order+1))//2)).to(device,dtype=torch.float32).requires_grad_()

        if slm_mode.lower() == 'random':
            slm_alphas = (slm_std * torch.rand(nframe, (zern_order*(zern_order+1))//2)).to(device,dtype=torch.float32)
            test_pattern = generate_zern_patterns(slm_alphas, zernike_basis, return_no_exp=False, device=device) #[1, nframe, 256, 256]

            if rand_zero_init:
                slm_alphas = torch.zeros_like(torch.abs(test_pattern)).to(device)
            else:
                slm_alphas = torch.randn_like(torch.abs(test_pattern)).to(device)

            del test_pattern
            gc.collect()
            torch.cuda.empty_cache()
        
        if slm_mode.lower() == 'focus':
            slm_alphas = gen_focus_sweep(focus_std=focus_std).to(device)
            print(colored('focus sweepa;lsdfj;alskdfj;laskdfj;alsdkfj;alsdkfj;lasldfjkas;dfljk', 'red'))
            info(slm_alphas, 'slm_alphas focus _sweep')

        if slm_mode.lower() == 'mlp':
            zern_net_basis = zern_basis_scalar * nn.Parameter(zernike_basis.permute(1, 2, 0).unsqueeze(0).repeat(1, 1, 1, 1), requires_grad=False)
            t_dim = 0
            phs_layers = mlp_hidden_layers
            in_dim = zern_net_basis.shape[-1]

            if mlp_act.lower() == 'leakyrelu':
                act_fn = nn.LeakyReLU(inplace=True)
            elif mlp_act.lower() == 'sigmoid':
                act_fn = nn.Sigmoid()
            else:
                print('Unknown MLP activation function! Must be leakyrelu or sigmoid!')
                raise NotImplementedError
            layers = []
            layers.append(nn.Linear(t_dim + in_dim, hidden_dim))
            for _ in range(phs_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn)

            layers.append(nn.Linear(hidden_dim, nframe))

            if mlp_init_scalar != 1:
                with torch.no_grad():
                    for single_layer in layers:
                        if isinstance(single_layer, nn.Linear):

                            single_layer.weight = torch.nn.Parameter(single_layer.weight.data * mlp_init_scalar)
                            single_layer.bias = torch.nn.Parameter(single_layer.bias.data * mlp_init_scalar)

            zern_net = nn.Sequential(*layers).to(device)
            slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)

        if slm_mode.lower() == 'dip':

            zern_net_basis = torch.rand(nframe, 1, img_size, img_size).to(device)   
            zern_net_basis.requires_grad = False
            zern_net  = ZernUNet(img_ch=1, output_ch=1, img_size=img_size, zern_order=zern_order, zern_insertion=zern_insertion, device=device)
            zern_net = zern_net.to(device)
            slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)

        if slm_mode.lower() == 'siren':
            zern_net_basis = torch.rand(nframe, 1, img_size, img_size).to(device)   
            zern_net_basis.requires_grad = False
            zern_net  = MultiSiren(img_size=img_size, nframes=nframe)
            zern_net = zern_net.to(device)
            slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)

        # generate the modulated image
        if slm_mode.lower() == 'zern':
            slm_patterns_unopt, _ = generate_zern_patterns(slm_alphas, zernike_basis, return_no_exp=True, device=device) #[1, nframe, 256, 256]
        else:
            slm_patterns_unopt = torch.exp(1j * slm_alphas)
            offset = (img_size - int(img_size / 1920 * 1080)) // 2
            slm_patterns_unopt = F.pad(slm_patterns_unopt[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)

        seed_torch(0)
        slm_alphas_zern_temp = (slm_std * torch.rand(nframe, (zern_order*(zern_order+1))//2)).to(device,dtype=torch.float32).requires_grad_()
        slm_patterns_unopt, _ = generate_zern_patterns(slm_alphas_zern_temp, zernike_basis, return_no_exp=True, device=device) #[1, nframe, 256, 256]
        slm_patterns_unopt = slm_patterns_unopt.clone().detach()

        if resume and use_modulation:
            print(torch.load(resume_ckpt_path).keys())
            slm_ckpt_path = resume_ckpt_path.replace('.pth', '_slm.pth')
            print(colored(torch.load(slm_ckpt_path).keys(), 'red'))

            if slm_mode == 'zern' or slm_mode == 'random' or slm_mode == 'focus':
                slm_alphas = torch.load(slm_ckpt_path)['slm_alphas'].to(device).requires_grad_()
            else:
                zern_net.load_state_dict(torch.load(resume_ckpt_path)['zern_mlp'])
                slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2).to(device).requires_grad_()
            print(colored(f'Loaded previously train SLM patterns from path: {slm_ckpt_path}', 'red'))     
            
        
        if 'control' not in slm_mode.lower():
            if slm_mode.lower() == 'zern' or slm_mode.lower() == 'random' or slm_mode.lower() == 'focus':
                slm_opt = torch.optim.Adam([slm_alphas], lr=init_lr)
            else:
                slm_opt = torch.optim.Adam(zern_net.parameters(), lr=init_lr)

            slm_scheduler = torch.optim.lr_scheduler.MultiStepLR(slm_opt, slm_lr_milestones, gamma=0.1)

        if learn_slm:
            if slm_mode.lower() == 'zern' or slm_mode.lower() == 'random':
                slm_alphas.requires_grad = True 
            elif slm_mode.lower() == 'mlp' or slm_mode.lower() == 'dip' or slm_mode.lower() == 'siren':
                for param in zern_net.parameters():
                    param.requires_grad = True

        else:
            if slm_mode.lower() == 'zern' or slm_mode.lower() == 'random' or slm_mode.lower() == 'focus':
                slm_alphas.requires_grad = False  
            elif slm_mode.lower() == 'mlp' or slm_mode.lower() == 'dip' or slm_mode.lower() == 'siren':
                for param in zern_net.parameters():
                    param.requires_grad = False

    mask_batch = gen_masks(width=img_size, grid_size=grid_size, mask_gaussian_std=mask_std, mask_gaussian_size=mask_size, DEVICE=torch.device('cuda'), vis=False)
    ang_to_unit = lambda x : ((x / np.pi) + 1) / 2

    # generate test aberration patterns, fixed during training
    seed_torch(22)
    abe_alphas_fixed_22 = (abe_std_low+abe_std_high)/2*torch.rand(grid_size**2, (zern_order*(zern_order+1))//2).to(device)
    abe_patterns_fixed_22 = generate_zern_patterns(abe_alphas_fixed_22, zernike_basis, device=device)
    seed_torch(42)
    abe_alphas_fixed_42 = (abe_std_low+abe_std_high)/2*torch.rand(grid_size**2, (zern_order*(zern_order+1))//2).to(device)
    abe_patterns_fixed_42 = generate_zern_patterns(abe_alphas_fixed_42, zernike_basis, device=device)
    
    #-------------------------------------------------------------------------------------
    #--- Training
    #-------------------------------------------------------------------------------------
    plot_mtf = False
    epoch_dir_new = None
    train_metrics = 0
    test_metrics, new_abe_metrics, new_slide_metrics = 0, 0, 0
    test_metrics_mean = 0
    new_abe_metrics_mean = 0
    new_slide_metrics_mean = 0
    best_metrics_mean = test_metrics_mean - 999
    best_metrics_mean_new_abe = new_abe_metrics - 999
    best_metrics_mean_new_slide = new_slide_metrics - 999
    slm_phs = 0
    epoch_bar = tqdm(range(num_epochs), desc='', disable=eval_only, dynamic_ncols=True, position=0, leave=True) 
    test_start_time = time.time()
    save_start_time = time.time()   
    best_save_model_list = []      
    beginning_time = time.time()
    for epoch in epoch_bar:
        model.train()
        epoch_data_visited = 0
        train_bar = tqdm(train_loader, desc='', disable=disable_inner_tqdm or eval_only, position=0, leave=True, dynamic_ncols=True)   
        for train_iter, train_batch in enumerate(train_bar, 1):
            epoch_data_visited += batch_size
            if (time.time() - save_start_time) > save_per_minutes * 60 or epoch_data_visited == batch_size and epoch == 0:
                num_hours_elapsed = int((time.time() - beginning_time) // 3600)
                save_start_time = time.time()

                if not eval_only:
                    if test_metrics_mean > best_metrics_mean or new_abe_metrics_mean > best_metrics_mean_new_abe or new_slide_metrics_mean > best_metrics_mean_new_slide:

                        if test_metrics_mean > best_metrics_mean:
                            best_metrics_mean = test_metrics_mean       
                        if new_abe_metrics_mean > best_metrics_mean_new_abe:
                            best_metrics_mean_new_abe = new_abe_metrics_mean
                        if new_slide_metrics_mean > best_metrics_mean_new_slide:
                            best_metrics_mean_new_slide = new_slide_metrics_mean

                        # save_model_name = f'{save_folder}_{num_hours_elapsed}h_epoch{epoch}_iter{epoch_data_visited-batch_size}_test_{test_metrics_mean:.2f}_new_abe_{new_abe_metrics_mean:.2f}_new_slide_{new_slide_metrics_mean:.2f}' 
                        save_model_name = f'Trained_{num_hours_elapsed}_hours_epoch{epoch}_iter{epoch_data_visited-batch_size}_PSNR_{test_metrics_mean:.2f}' 
                        model_out_path = f'{save_ckpt_dir}/pth/{save_model_name}.pth'

                        if sim and use_modulation:
                            outs = {'epoch': epoch + 1, 'arch': model, 'state_dict': model.state_dict(),
                                'name': 0, 'optimizer' : optimizer.state_dict(), 'slm_alphas': slm_alphas, 
                                }
                            if slm_mode.lower() == 'mlp' or slm_mode.lower() == 'dip' or slm_mode.lower() == 'siren':
                                outs['zern_mlp'] = zern_net.state_dict()
                        else:
                            outs = {'epoch': epoch + 1, 'arch': model, 'state_dict': model.state_dict(),
                                'name': 0, 'optimizer' : optimizer.state_dict()}
                                                       
                        torch.save(outs, model_out_path)

                        if use_modulation:   
                            if slm_mode.lower() == 'zern':
                                _, slm_temp = generate_zern_patterns(slm_alphas, zernike_basis, return_no_exp=True, device=device) #[1, nframe, 256, 256]
                            else:
                                if slm_mode.lower() == 'random' or slm_mode.lower() == 'focus':
                                    slm_temp = slm_alphas
                                else:
                                    slm_temp = zern_net(zern_net_basis).permute(0, 3, 1, 2)

                            slm_out_path = f'{save_ckpt_dir}/pth/{save_model_name}_slm.pth'
                            torch.save({'slm_alphas': slm_alphas, 'slm_phs': slm_temp}, slm_out_path)
                            mat_out_path = f'{save_ckpt_dir}/mat/{save_model_name}.mat'
                            sio.savemat(mat_out_path, {'slm_phs':slm_temp.detach().cpu().numpy()}) 

                        print(colored(f"Epoch {epoch} data {epoch_data_visited-batch_size} ===> Checkpoint Saved At: {model_out_path} ...", 'red'))

                        if sim and overwrite_ckpts:
                            if len(best_save_model_list) > 0:
                                for temp_path in best_save_model_list:
                                    if os.path.exists(temp_path):
                                        if temp_path.split('.')[-1] in ['pth', 'mat']:
                                            os.remove(temp_path)
                        if use_modulation:   
                            best_save_model_list = [model_out_path, slm_out_path, mat_out_path]  
                        else:
                            best_save_model_list = [model_out_path]       

                    else:
                        if len(best_save_model_list) > 0:
                            print(colored(f'Epoch {epoch} data {epoch_data_visited-batch_size} Current PSNR is {test_metrics_mean}. Best model is still the one saved at {best_save_model_list[-1]}', 'red'))
                        else:
                            print(colored(f"Epoch {epoch} data {epoch_data_visited-batch_size}    ===> Not Saving Checkpoint ...", 'red'))

            if eval_only or (time.time() - test_start_time) > test_per_minutes * 60 or epoch_data_visited == batch_size:

                plot_mtf = True
                test_start_time = time.time()
                if verbose:
                    print(f"          ===> Saving Test Results ...")

                epoch_dir = f'{save_dir}/epochs/epoch_{epoch}_data_visited_{epoch_data_visited-batch_size}'

                if epoch_dir_new is not None:
                    last_epoch_dir_renamed = f'{epoch_dir_new}_metrics_psnr_{test_metrics_mean:.4f}_ssim_{test_metrics_mean_ssim:.4f}'
                    os.rename(epoch_dir_new, last_epoch_dir_renamed)
                    print(colored(f'Epoch {epoch} data {epoch_data_visited-batch_size} ===> Renamed {epoch_dir_new} to {last_epoch_dir_renamed}', 'red'))

                epoch_subdir_list = [epoch_dir]
                for epoch_subdir in epoch_subdir_list:
                    if not os.path.exists(epoch_subdir):
                        os.mkdir(epoch_subdir)

                # epoch_img_dir = f'{save_dir}/epochs_img/e{epoch}_d{epoch_data_visited-batch_size}' 
                epoch_img_dir = epoch_dir
                if not os.path.exists(epoch_img_dir): 
                    os.mkdir(epoch_img_dir)

                # Test log
                test_metrics_sum = 0
                test_metrics_sum_ssim = 0
                count = 0
                test_bar = tqdm(test_loader, desc='Test', disable=not verbose, position=0, leave=False, dynamic_ncols=True)   

                seed_torch(42)
                if True:
                    for test_iter, test_batch in enumerate(test_bar, 1):
                        if True:
                            seed_torch(test_iter+42)
                            if not sim:
                                target, sample = test_batch[0].to(device), test_batch[1].to(device)
                                recon = model(sample)
                            else:
                                if dataset.lower() == 'fashion':
                                    test_batch = test_batch[0]
                                target = test_batch.to(device) #[1, 1, 256, 256]
                                if low_pass_img:
                                    target = tgm.image.gaussian_blur(target, (low_pass_level, low_pass_level), (low_pass_level, low_pass_level))
                                
                                abe_std = torch.FloatTensor(1).uniform_(abe_std_low, abe_std_high).to(device)
                                abe_alphas = abe_std*torch.rand(grid_size**2, (zern_order*(zern_order+1))//2).to(device)
                                abe_patterns = generate_zern_patterns(abe_alphas, zernike_basis, device=device)
                                
                                if random_abe:
                                    rand_std = torch.FloatTensor(1).uniform_(rand_std_low, rand_std_high).to(device)
                                    abe_patterns = generate_rand_pattern_like(abe_patterns, std=rand_std, device=device)
                                
                                abe_psfs = gen_psf(abe_patterns)
                                y_zero = conv_psf(target, abe_psfs, mask=mask_batch) # [nbatch, 1, 1, 256, 256]
                                if use_modulation:
                                    if slm_mode.lower() == 'zern':
                                        slm_patterns, slm_phs = generate_zern_patterns(slm_alphas, zernike_basis, return_no_exp=True, device=device) #[1, nframe, 256, 256]
                                    else:
                                        if slm_mode.lower() == 'random' or slm_mode.lower() == 'focus':
                                            slm_phs = slm_alphas
                                        if slm_mode.lower() == 'mlp' or slm_mode.lower() == 'dip' or slm_mode.lower() == 'siren':
                                            slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)
                                            slm_phs = slm_alphas

                                        slm_patterns = torch.exp(1j * slm_alphas)
                                        offset = (img_size - int(img_size / 1920 * 1080)) // 2
                                        slm_patterns = F.pad(slm_patterns[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)

                                    mod_psfs = gen_psf(slm_patterns.permute(1, 0, 2, 3) * abe_patterns) # nframe, grid**2, 256, 256]
                                    y_mod = conv_psf(target, mod_psfs, mask=mask_batch)  # nframe, 1, 256, 256]
                                    sample = torch.cat((y_zero, y_mod), dim=1) # [nbatch, nframe+1, 1, 256, 256]

                                else:
                                    sample = y_zero
                                
                                recon = model(sample)

                            test_metrics = psnr(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
                            test_metrics_ssim = ssim(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
                        
                        if not eval_only:
                            epoch_bar.set_description(f"[Epoch {epoch}] ===> Train PSNR: {train_metrics:.4f}, Test PSNR: {test_metrics:.4f}, New Abe PSNR: {new_abe_metrics:.4f}")
                            epoch_bar.refresh()
                        test_metrics_sum += test_metrics
                        test_metrics_sum_ssim += test_metrics_ssim
                        count = count + 1

                        img_list = [target[0].unsqueeze(0).detach().cpu(), sample[0, 0, ...].unsqueeze(0).detach().cpu(), recon[0].unsqueeze(0).detach().cpu()]
                        # test_save_path =  f'{epoch_dir}/TestSet_epoch_{epoch}_iter_{epoch_data_visited-batch_size}_psnr_{test_metrics:.4f}.png'
                        # test_save_path_no_psnr = f'{epoch_dir}/TestSet_epoch_{epoch}_iter_{epoch_data_visited-batch_size}_psnr_{test_metrics:.4f}.png'
                        # test_save_path_imgs =  f'{epoch_img_dir}/ztest_epoch{epoch}_idx{test_iter}_psnr{test_metrics:.4f}'

                        if test_iter <= 40:
                            pass

                            # test_save_path = save_captioned_imgs(img_list, caption_list=['GT', None, 'Recon'], is_torch_tensor=True, rescale=False, flip=True, grayscale=False, 
                            # save_path=test_save_path)
                            # log_image(target[0].unsqueeze(0), 'Test-Target', f'{epoch_dir_vis}/test_epoch{epoch}_idx{test_iter}_target', log_wandb=False)
                            # log_image(sample[0, 0, ...].unsqueeze(0), 'Test-Zero', f'{epoch_dir_vis}/test_epoch{epoch}_idx{test_iter}_zero', log_wandb=False)
                            # log_image(recon[0].unsqueeze(0), 'Test-Recon', f'{epoch_dir_vis}/test_epoch{epoch}_idx{test_iter}_recon', log_wandb=False)

                        if test_iter == 1:

                            test_save_path = save_captioned_imgs(img_list, caption_list=[None, None, None], is_torch_tensor=True, rescale=False, flip=True, grayscale=False, 
                            save_path=f'{epoch_dir}/TestSet_GT_Measurement_Recon_PSNR_{test_metrics:.4f}.png')
                            
                            save_image(target[0].unsqueeze(0).detach().cpu(), f'{epoch_dir}/TestSet_GT.png', normalize=False, scale_each=False)
                            save_image(recon[0].unsqueeze(0).detach().cpu(), f'{epoch_dir}/TestSet_Recon_PSNR_{test_metrics:.4f}.png', normalize=False, scale_each=False)


                            if use_wandb:
                                wandb.log({"TestSet: GT vs. Measurement vs. Recon": wandb.Image(test_save_path)}, step=train_iter)

                            if sim:
                                phs_vis_num = max(nframe, 1)  
                                abe_ang = ang_to_unit(torch.angle(abe_patterns.permute(1, 0, 2, 3)[:phs_vis_num])) # spatially varying, thus 4 aberration phases!
 
                                # save_image(sample[0][0].unsqueeze(0), f'{save_dir}/slm/zero_epoch{epoch}_idx{test_iter}_psnr{test_metrics:.4f}.png', normalize=True, scale_each=False)
                                zero_save_dir = f'{epoch_dir}/TestSet_Unmodulated_Measurement.png'
                                save_image(sample[0][0].unsqueeze(0), zero_save_dir, normalize=True, scale_each=False)
                                # save_image(abe_ang, f'{save_dir}/slm/Abe_{save_folder}.png', nrow=4, normalize=True, scale_each=False)
                                save_image(abe_ang, f'{epoch_dir}/Aberration_Phase.png', nrow=4, normalize=True, scale_each=False)

                                if use_wandb:
                                    wandb.log({"Wavefront Error Due To Scattering": wandb.Image(f'{epoch_dir}/Aberration_Phase.png')}, step=train_iter)
                                    # wandb.log({"Zero": wandb.Image(zero_save_dir)}, step=train_iter)
                                    # log_image(abe_ang[0][0].squeeze().unsqueeze(0).unsqueeze(0), 'Abe-Phase', f'{epoch_dir_abe}/abe_ang_epoch{epoch}_idx{test_iter}', log_wandb=False)

                                if use_modulation:
                                    # sio.savemat(f'{save_dir}/slm/zmat_epoch{epoch}_SLMphs_no_exp_{save_folder}.mat', {'slm':slm_phs.detach().cpu().numpy()}) 
                                    sio.savemat(f'{epoch_dir}/Wavefront_Modulations.mat', {'slm':slm_phs.detach().cpu().numpy()}) 

                                    slm_ang = ang_to_unit(torch.angle(slm_patterns.permute(1, 0, 2, 3)[:phs_vis_num]))


                                    # save_image(sample[0][1:phs_vis_num+1], f'{save_dir}/iters/sample_epoch{epoch}.png', nrow=4, normalize=True, scale_each=False)
                                    save_image(sample[0][1:phs_vis_num+1], f'{epoch_dir}/Modulated_Measurements.png', nrow=4, normalize=True, scale_each=False)
                                    
                                    # save_image(slm_ang, f'{save_dir}/slm/SLM_epoch{epoch}_{save_folder}.png', nrow=4, normalize=True, scale_each=False)
                                    save_image(slm_ang, f'{epoch_dir}/Wavefront_Modulations.png', nrow=4, normalize=True, scale_each=False)
                                    # save_image(torch.log(slm_ang), f'{epoch_dir}/Log-SLM_epoch{epoch}_{save_folder}.png', nrow=4, normalize=True, scale_each=False)

                                    if use_wandb:
                                        wandb.log({"Modulation Patterns": wandb.Image(f'{epoch_dir}/Wavefront_Modulations.png')}, step=train_iter)
                                        wandb.log({"Modulated Measurements": wandb.Image(f'{epoch_dir}/Modulated_Measurements.png')}, step=train_iter)    

                    del img_list, target, sample, recon
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    test_metrics_mean = test_metrics_sum / count
                    test_metrics_mean_ssim = test_metrics_sum_ssim / count

                    # with open(f'{epoch_dir_new}/metrics_test_psnr_{test_metrics_mean}_ssim_{test_metrics_mean_ssim}.txt', "w") as file:
                    #     file.write("Please refer to the filename.")
                    
                    if use_wandb:
                        wandb.log({'Test-PSNR': test_metrics_mean}, step=train_iter)
                        wandb.log({'Test-SSIM': test_metrics_mean_ssim}, step=train_iter)
                        wandb.log({'Data-Visited': (epoch_data_visited-batch_size)}, step=train_iter)

                    if eval_only:
                        print('Test PSNR: ', test_metrics_mean, 'Test SSIM: ', test_metrics_mean_ssim)

                    # Unseen Abe log
                    if not sim:
                        new_abe_metrics_sum = 0
                        new_abe_metrics_sum_ssim = 0
                        count = 0
                        new_abe_bar = tqdm(new_abe_loader, desc='', disable=not verbose, position=0, leave=False, dynamic_ncols=True)   
                        for new_abe_iter, new_abe_batch in enumerate(new_abe_bar, 1):
                            if True:
                                target, sample = new_abe_batch[0].to(device), new_abe_batch[1].to(device)
                                recon = model(sample)
                                new_abe_metrics = psnr(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
                                new_abe_metrics_ssim = ssim(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
                            
                            if not eval_only:
                                epoch_bar.set_description(f"[Epoch {epoch}] ===> Train PSNR: {train_metrics:.4f}, Test PSNR: {test_metrics:.4f}, New Abe PSNR: {new_abe_metrics:.4f}")
                                epoch_bar.refresh()
                            img_list = [target[0].unsqueeze(0), sample[0, 0, ...].unsqueeze(0), recon[0].unsqueeze(0)]
                            new_abe_save_path = f'outs/new_abe.png' if epoch % save_frequency != 0 else f'{epoch_dir}/new_abe_epoch{epoch}_idx{new_abe_iter}_psnr{new_abe_metrics:.4f}.png'
                            new_abe_save_path_images = f'outs/new_abe' if epoch % save_frequency != 0 else f'{epoch_img_dir}/znew_abe_epoch{epoch}_idx{new_abe_iter}_psnr{new_abe_metrics:.4f}'
                            new_abe_save_path = save_captioned_imgs(img_list, caption_list=['truth', 'sample', 'recon'], is_torch_tensor=True, rescale=False, flip=True, grayscale=False, 
                                                    save_path=new_abe_save_path, save_imgs_path=new_abe_save_path_images)
                            if new_abe_iter == 1:
                                if use_wandb:
                                    wandb.log({'Unseen-Abe': wandb.Image(new_abe_save_path)}, step=train_iter)              
                                if epoch == 0 and use_modulation:
                                    save_image(transforms.functional.vflip(sample[0][1:]), f'{save_dir}/modulated_samples.png', nrow=4, normalize=True, scale_each=False)
                                    save_image(transforms.functional.vflip(sample[0][1:]), f'{epoch_dir}/new_abe_epoch{epoch}_idx{new_abe_iter}_samples.png', nrow=4, normalize=True, scale_each=False)
                                    if use_wandb:
                                
                                        wandb.log({"Modulated Samples": wandb.Image(f'{save_dir}/modulated_samples.png')}, step=train_iter)              

                            new_abe_metrics_sum += new_abe_metrics
                            new_abe_metrics_sum_ssim += new_abe_metrics_ssim
                            count = count + 1
  
                        new_abe_metrics_mean = new_abe_metrics_sum / count
                        new_abe_metrics_mean_ssim = new_abe_metrics_sum_ssim / count
                        if use_wandb:
                                    
                            wandb.log({'Unseen-Abe-PSNR-Mean': new_abe_metrics_mean}, step=train_iter)
                            wandb.log({'Unseen-Abe-SSIM-Mean': new_abe_metrics_mean_ssim}, step=train_iter)
                        if eval_only:
                            print('New Abe PSNR: ', new_abe_metrics_mean, 'New Abe SSIM: ', new_abe_metrics_mean_ssim)  

                        del target, sample, recon
                        del img_list
                        torch.cuda.empty_cache()
                        gc.collect()
                        new_slide_metrics_sum = 0
                        new_slide_metrics_sum_ssim = 0
                        count = 0
                        new_slide_bar = tqdm(new_slide_loader, desc='', disable=not verbose, position=0, leave=False, dynamic_ncols=True)   
                        for new_slide_iter, new_slide_batch in enumerate(new_slide_bar, 1):
                            if True:
                                target, sample = new_slide_batch[0], new_slide_batch[1].to(device)
                                recon = model(sample)
                                target = target.to(device)
                                new_slide_metrics = psnr(torch.clamp(crop_ROI(recon), 0, 1), crop_ROI(target)).item()
                                new_slide_metrics_ssim = ssim(torch.clamp(crop_ROI(recon), 0, 1), crop_ROI(target)).item()
                            
                            if not eval_only:
                                epoch_bar.set_description(f"[Epoch {epoch}] ===> Train PSNR: {train_metrics:.4f}, Test PSNR: {test_metrics:.4f}, New Abe PSNR: {new_slide_metrics:.4f}")
                                epoch_bar.refresh()
                            img_list = [target[0].unsqueeze(0).detach().cpu(), sample[0, 0, ...].unsqueeze(0).detach().cpu(), recon[0].unsqueeze(0).detach().cpu()]
                            new_slide_save_path = f'outs/new_slide.png' if epoch % save_frequency != 0 else f'{epoch_dir}/new_slide_epoch{epoch}_idx{new_slide_iter}_psnr{new_slide_metrics:.4f}.png'
                            new_slide_save_path_images = f'outs/new_slide' if epoch % save_frequency != 0 else f'{epoch_img_dir}/znew_slide_epoch{epoch}_idx{new_slide_iter}_psnr{new_slide_metrics:.4f}'
                            new_slide_save_path = save_captioned_imgs(img_list, caption_list=['truth', 'sample', 'recon'], is_torch_tensor=True, rescale=False, flip=True, grayscale=False, 
                                                    save_path=new_slide_save_path, save_imgs_path=new_slide_save_path_images)
                            if new_slide_iter == 1:
                                if use_wandb:
                                    wandb.log({'Unseen-Slide': wandb.Image(new_slide_save_path)}, step=train_iter)              

                                if epoch == 0 and use_modulation:
                                    save_image(transforms.functional.vflip(sample[0][1:]), f'{save_dir}/modulated_samples.png', nrow=4, normalize=True, scale_each=False)
                                    save_image(transforms.functional.vflip(sample[0][1:]), f'{epoch_dir}/new_slide_epoch{epoch}_idx{new_slide_iter}_samples.png', nrow=4, normalize=True, scale_each=False)
                                    if use_wandb:
                                        wandb.log({"Modulated Samples": wandb.Image(f'{save_dir}/modulated_samples.png')}, step=train_iter)              

                            new_slide_metrics_sum += new_slide_metrics
                            new_slide_metrics_sum_ssim += new_slide_metrics_ssim
                            count = count + 1

                        new_slide_metrics_mean = new_slide_metrics_sum / count
                        new_slide_metrics_mean_ssim = new_slide_metrics_sum_ssim / count
                        if use_wandb:
                            wandb.log({'Unseen-slide-PSNR-Mean': new_slide_metrics_mean}, step=train_iter)
                            wandb.log({'Unseen-slide-SSIM-Mean': new_slide_metrics_mean_ssim}, step=train_iter)

                        if eval_only:
                            print('New Slide PSNR: ', new_slide_metrics_mean, 'New Slide SSIM: ', new_slide_metrics_mean_ssim)

                        del target, sample, recon, img_list
                        gc.collect()
                        torch.cuda.empty_cache()

                    seed_torch(epoch * len(train_loader) + train_iter) 
                    model.train()

            if eval_only:
                print('Just finished evaluation. Exiting ...')
                print(colored(f'Saved at {epoch_dir}', 'red'))
                if use_wandb:
                    run.finish()
                    wandb.finish()
                return
                        
            if not sim:
                target, sample = train_batch[0].to(device), train_batch[1].to(device)
                recon = model(sample)
                loss = nn.MSELoss()(recon, target)  
            else:
                if dataset.lower() == 'fashion':
                    train_batch = train_batch[0]

                target = train_batch.to(device) #[1, 1, 256, 256]
                if low_pass_img:
                    target = tgm.image.gaussian_blur(target, (low_pass_level, low_pass_level), (low_pass_level, low_pass_level))

                # generate the zero image
                abe_std = torch.FloatTensor(1).uniform_(abe_std_low, abe_std_high).to(device)
                abe_alphas = abe_std*torch.rand(grid_size**2, (zern_order*(zern_order+1))//2).to(device)
                abe_patterns = generate_zern_patterns(abe_alphas, zernike_basis, device=device)  # [1, grid**2, 256, 256]

                if random_abe:
                    rand_std = torch.FloatTensor(1).uniform_(rand_std_low, rand_std_high).to(device)
                    abe_patterns = generate_rand_pattern_like(abe_patterns, std=rand_std, device=device)   

                abe_psfs = gen_psf(abe_patterns)  # [1, grid**2, 256, 256]
                y_zero = conv_psf(target, abe_psfs, mask=mask_batch) # [1, 1, 256, 256]

                if use_modulation:
                    # generate the modulated image
                    if slm_mode.lower() == 'zern':
                        slm_patterns = generate_zern_patterns(slm_alphas, zernike_basis, device=device) #[1, nframe, 256, 256]
                    else:
                        if slm_mode.lower() == 'random' or slm_mode.lower() == 'focus':
                            pass
                        if slm_mode.lower() == 'mlp' or slm_mode.lower() == 'dip' or slm_mode.lower() == 'siren':
                            slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)  
                        slm_patterns = torch.exp(1j * slm_alphas)
                        offset = (img_size - int(img_size / 1920 * 1080)) // 2
                        slm_patterns = F.pad(slm_patterns[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)

                    mod_psfs = gen_psf(slm_patterns.permute(1, 0, 2, 3) * abe_patterns) # nframe, grid**2, 256, 256]
                    y_mod = conv_psf(target, mod_psfs, mask=mask_batch)  # nframe, 1, 256, 256]
                    sample = torch.cat((y_zero, y_mod), dim=1) # [1, nframe+1, 1, 256, 256]

                else:
                    sample = y_zero

                recon = model(sample)
                loss = nn.MSELoss()(recon, target) 

                if use_wandb:
                    wandb.log({'Train-Loss': nn.MSELoss()(recon, target).item()}, step=train_iter)
                    wandb.log({'Train-Iter': train_iter}, step=train_iter)

                if use_modulation:
                    slm_loss = torch.min((target - sample.squeeze(2))**2, dim=1)[0].mean()
                    loss += slm_loss_weight * slm_loss

                    tv_loss = tv(slm_patterns).mean()
                    loss += tv_weight * tv_loss

                    if use_wandb:
                        wandb.log({'Train-TV': tv_weight * tv_loss}, step=train_iter)

                    if mtf_loss:
                        
                        def compute_normalized_MTF(input_slm_patterns):
                            system_final = input_slm_patterns.permute(1, 0, 2, 3) * abe_patterns_fixed_42
                            system_final = system_final.squeeze(1)
                            F_slm = torch.fft.fft2(system_final)
                            power_spectrum = F_slm.abs() ** 2
                            OTF = torch.fft.fftshift(torch.fft.ifft2(power_spectrum))
                            MTF = torch.abs(OTF)
                            return MTF   
                        
                        MTF = compute_normalized_MTF(slm_patterns)
                        MTF_maxpooled_opt = (MTF / torch.amax(MTF, dim=(1, 2), keepdim=True)).max(dim=0)[0]
                        normalized_row_opt = MTF_maxpooled_opt.squeeze()[..., len(MTF_maxpooled_opt)//2, :]

                        MTF_unopt = compute_normalized_MTF(slm_patterns_unopt)
                        MTF_maxpooled_unopt = (MTF_unopt / torch.amax(MTF_unopt, dim=(1, 2), keepdim=True)).max(dim=0)[0]
                        normalized_row_unopt = MTF_maxpooled_unopt.squeeze()[..., len(MTF_maxpooled_unopt)//2, :]  


                        if plot_mtf:
                            plot_mtf = False                            
                            def plot_MTF(opt_row, unopt_row, log_scale=None, use_wandb=True, title=''):
                                plt.figure(figsize=(10, 6))
                                plt.plot(opt_row.detach().cpu().numpy(), label='Opt MTF Central Row', color='red')
                                plt.plot(unopt_row.detach().cpu().numpy(), label='Unopt MTF Central Row', color='blue')
                                plt.legend()
                                plt.xlabel('Column Index')
                                plt.ylabel('Normalized MTF')
                                if log_scale is not None:
                                    plt.yscale('log',base=log_scale) 
                                plt.title(f'{title} MTF (Log {log_scale})')
                                plt.grid(True)
                                # plt.savefig(f'{save_dir}/mtf/{title}_log_{log_scale}_mtf_epoch{epoch}_idx{train_iter}.png')
                                mtf_save_dir = f'{epoch_dir}/{title}.png'
                                plt.savefig(mtf_save_dir)
                                # plt.savefig(f'{epoch_dir_mtf}/{title}1D_log{log_scale}_mtf_opt_epoch{epoch}_idx{train_iter}')
                                if use_wandb:
                                    wandb.log({f"{title}": wandb.Image(mtf_save_dir)}, step=train_iter)
                                plt.close()

                            plot_MTF(log_scale=10, opt_row=normalized_row_opt, unopt_row=normalized_row_unopt, title='Modulation_Transfer_Function', use_wandb=use_wandb)

                        if mtf_mode.lower() == 'var':
                            var = torch.var(MTF.max(dim=0)[0])
                        elif mtf_mode.lower() == 'ratio':
                            var = (torch.amax(MTF, dim=(1, 2)) / torch.amin(MTF, dim=(1, 2))).mean()
                        elif mtf_mode.lower() == 'one':
                            var = (1 - MTF.max(dim=0)[0]).sum()
                        else:
                            assert False, 'mtf_mode not recognized! Must be var or ratio!'   
                        loss += mtf_weight * var
                        if use_wandb:        
                            wandb.log({'Train-MTF': mtf_weight * var}, step=train_iter)
    
            loss /= accum_batch
            loss.backward()
            if train_iter % accum_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

                if learn_slm and sim:
                    if 'control' not in slm_mode.lower():
                        slm_opt.step()
                        slm_opt.zero_grad()

            train_metrics = psnr(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
            train_metrics_ssim = ssim(torch.clamp(crop_ROI(recon.detach()), 0, 1), crop_ROI(target.detach())).item()
            epoch_bar.set_description(f"[Epoch {epoch}] Data Visited {epoch_data_visited-batch_size} ===> Train PSNR: {train_metrics:.4f}, Test PSNR: {test_metrics:.4f}, New Abe PSNR: {new_abe_metrics:.4f}")
            epoch_bar.refresh()
            if use_wandb:
                wandb.log({'Train-PSNR': train_metrics}, step=train_iter)  
                wandb.log({'Train-SSIM': train_metrics_ssim}, step=train_iter)

            if not sim and train_iter == 1:
                img_list = [target[0].unsqueeze(0).detach().cpu(), sample[0, 0, ...].unsqueeze(0).detach().cpu(), recon[0].unsqueeze(0).detach().cpu()]
                train_save_path = save_captioned_imgs(img_list, caption_list=['truth', 'sample', 'recon'], is_torch_tensor=True, rescale=False, flip=True,
                                        save_path=f'outs/train.png', grayscale=False)
                if use_wandb:
                    wandb.log({"Train-Recon": wandb.Image(train_save_path)}, step=train_iter)

                del sample, target, recon
                torch.cuda.empty_cache()

            if sim:
                scheduler.step()
                if learn_slm:
                    if 'control' not in slm_mode.lower():
                        slm_scheduler.step()
       
        if not sim and num_epochs > 1:
            scheduler.step()
            del sample, target, recon
            torch.cuda.empty_cache()
            gc.collect()

    if use_wandb:
        run.finish()
