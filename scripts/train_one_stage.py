import argparse
import os
from typing import Dict, Any, Optional

def get_config_file(scene: str) -> str:
    """Map scene name to corresponding config file."""
    
    # Dictionary mapping scene names to config files
    SCENE_CONFIG_MAPPING = {
        'roomy': 'blender_ngp_yobo_roomy',
        'kettle': 'transient_simulation_ngp_yobo_kettle',
        'statue': 'transient_simulation_ngp_yobo_statue',
        'statue_old': 'transient_simulation_ngp_yobo_statue_old',
        'statue_tnerf_old': 'transient_simulation_ngp_yobo_statue_tnerf_old',
        'house_red': 'transient_simulation_ngp_yobo_house_red',
        'house_green': 'transient_simulation_ngp_yobo_house_green',
        'house_blue': 'transient_simulation_ngp_yobo_house_blue',
        'spheres': 'transient_simulation_ngp_yobo_spheres',
        'spheres_old': 'transient_simulation_ngp_yobo_spheres_old',
        'spheres_tnerf_old': 'transient_simulation_ngp_yobo_spheres_tnerf_old',
        'statue_fwp': 'transient_simulation_ngp_yobo_statue_fwp',
        'kettle_fwp': 'transient_simulation_ngp_yobo_kettle_fwp',
        'globe_fwp': 'transient_simulation_ngp_yobo_globe_fwp',
        'house_fwp': 'transient_simulation_ngp_yobo_house_fwp',
        'spheres_fwp': 'transient_simulation_ngp_yobo_spheres_fwp',
        'statue_tnerf': 'transient_simulation_ngp_yobo_statue_tnerf',
        'kettle_tnerf': 'transient_simulation_ngp_yobo_kettle_tnerf',
        'spheres_tnerf': 'transient_simulation_ngp_yobo_spheres_tnerf',
        'spheres_itof': 'transient_simulation_ngp_yobo_spheres_itof',
        'spheres_steady_state': 'transient_simulation_ngp_yobo_spheres_steady_state',
        'kettle_views_removed': 'transient_simulation_ngp_yobo_kettle_views_removed',
        'peppers': 'transient_simulation_ngp_yobo_peppers',
        'peppers_itof': 'transient_simulation_ngp_yobo_peppers_itof',
        'peppers_steady_state': 'transient_simulation_ngp_yobo_peppers_steady_state',
        'cornell_itof': 'transient_simulation_ngp_yobo_cornell_itof',
        'cornell_steady_state': 'transient_simulation_ngp_yobo_cornell_steady_state',
        'kitchen_itof': 'transient_simulation_ngp_yobo_kitchen_itof',
        'kitchen_steady_state': 'transient_simulation_ngp_yobo_kitchen_steady_state',
        'pots_itof': 'transient_simulation_ngp_yobo_pots_itof',
        'pots_steady_state': 'transient_simulation_ngp_yobo_pots_steady_state',
        'globe_tnerf': 'transient_simulation_ngp_yobo_globe_tnerf',
        'house_tnerf': 'transient_simulation_ngp_yobo_house_tnerf',
        'house': 'transient_simulation_ngp_yobo_house',
        'house_old': 'transient_simulation_ngp_yobo_house_old',
        'house_itof': 'transient_simulation_ngp_yobo_house_itof',
        'house_steady_state': 'transient_simulation_ngp_yobo_house_steady_state',
        'globe': 'transient_simulation_ngp_yobo_globe',
        'globe_steady_state': 'transient_simulation_ngp_yobo_globe_steady_state',
        'kitchen': 'transient_simulation_ngp_yobo_kitchen',
        'peppers_tnerf': 'transient_simulation_ngp_yobo_peppers_tnerf',
        'pots_tnerf': 'transient_simulation_ngp_yobo_pots_tnerf',
        'cornell_tnerf': 'transient_simulation_ngp_yobo_cornell_tnerf',
        'kitchen_tnerf': 'transient_simulation_ngp_yobo_kitchen_tnerf',
        'kitchen_stopgrads': 'transient_simulation_ngp_yobo_kitchen_stopgrads',
        'kitchen_rawnerf_tune': 'transient_simulation_ngp_yobo_kitchen_rawnerf_tune',
        'peppers_fwp': 'transient_simulation_ngp_yobo_peppers_fwp',
        'peppers_itof': 'transient_simulation_itof_ngp_yobo_peppers',
        'peppers_steady': 'transient_simulation_steady_ngp_yobo_peppers',
        'pots_fwp': 'transient_simulation_ngp_yobo_pots_fwp',
        'pots_itof': 'transient_simulation_itof_ngp_yobo_pots',
        'cornell_fwp': 'transient_simulation_ngp_yobo_cornell_fwp',
        'cornell_fwp_dataset': 'transient_simulation_ngp_yobo_cornell_fwp_dataset',
        'cornell_itof': 'transient_simulation_itof_ngp_yobo_cornell',
        'cornell_flash_dense': 'transient_simulation_ngp_yobo_cornell_flash_dense',
        'cornell_elevated': 'transient_simulation_ngp_yobo_cornell_elevated',
        'pots': 'transient_simulation_ngp_yobo_pots',
        'cornell_flash_dense_multibounce': 'transient_simulation_ngp_yobo_cornell_flash_dense_multibounce',
        'cornell': 'transient_simulation_ngp_yobo_cornell',
        'cornell_flash_rawnerf': 'transient_simulation_ngp_yobo_cornell_flash_rawnerf',
        'cornell_flash_white_walls': 'transient_simulation_ngp_yobo_cornell_flash_white_walls',
        'cornell_flash_normal': 'transient_simulation_ngp_yobo_cornell_flash',
        'cornell_flash_debug': 'transient_simulation_ngp_yobo_cornell_flash_debug',
        'cornell_flash_mask': 'transient_simulation_ngp_yobo_cornell_flash_mask',
        'cornell_flash_distortion': 'transient_simulation_ngp_yobo_cornell_flash_distortion',
        'roomy_mirror': 'blender_ngp_yobo_roomy_mirror',
        'roomy_point_light': 'blender_ngp_yobo_roomy_point_light',
        'roomy_point_light_spotlight': 'blender_ngp_yobo_roomy_point_light_spotlight',
        'roomy_point_light_spotlight_shiny': 'blender_ngp_yobo_roomy_point_light_spotlight_shiny',
        'roomy_point_light_no_ambient': 'blender_ngp_yobo_roomy_point_light_no_ambient',
        'roomy_point_light_no_ambient_shiny': 'blender_ngp_yobo_roomy_point_light_no_ambient_shiny',
        'roomy_point_light_no_ambient_shiny_direct': 'blender_ngp_yobo_roomy_point_light_no_ambient_shiny_direct',
        'lego_pano': 'blender_ngp_yobo_lego',
        'lego_box': 'blender_ngp_yobo_lego_box3',
        'lego': 'nerf_ngp_yobo_lego',
        'hotdog': 'nerf_ngp_yobo_hotdog',
        'armadillo': 'nerf_ngp_yobo_armadillo',
        'ficus': 'nerf_ngp_yobo_ficus',
        'gnome': 'orb_ngp_yobo_gnome',
        'pitcher': 'orb_ngp_yobo_pitcher',
        'cactus': 'orb_ngp_yobo_cactus',
        'teapot': 'orb_ngp_yobo_teapot',
        'castel': 'neilf_ngp_yobo_castel',
        'obj_02_egg': 'open_ngp_yobo_egg',
        'obj_04_stone': 'open_ngp_yobo_stone',
        'obj_05_bird': 'open_ngp_yobo_bird',
        'obj_17_box': 'open_ngp_yobo_box',
        'obj_26_pumpkin': 'open_ngp_yobo_pumpkin',
        'obj_29_hat': 'open_ngp_yobo_hat',
        'obj_35_cup': 'open_ngp_yobo_cup',
        'obj_36_sponge': 'open_ngp_yobo_sponge',
        'obj_42_banana': 'open_ngp_yobo_banana',
        'obj_48_bucket': 'open_ngp_yobo_bucket',
        'glossy_bunny': 'glossy_bunny_yobo',
        'glossy_vase': 'glossy_vase_yobo',
        'nero_angel': 'nero_ngp_yobo_angel',
        'nero_tbell': 'nero_ngp_yobo_tbell',
        'nero_bell': 'nero_ngp_yobo_bell',
        'nero_cat': 'nero_ngp_yobo_cat',
        'nero_horse': 'nero_ngp_yobo_horse',
        'nero_luyu': 'nero_ngp_yobo_luyu',
        'nero_potion': 'nero_ngp_yobo_potion',
        'nero_teapot': 'nero_ngp_yobo_teapot',
        'neilf_cat': 'neilf_cat_yobo',
    }
    
    # Return matching config file or raise error if scene not found
    if scene in SCENE_CONFIG_MAPPING:
        return SCENE_CONFIG_MAPPING[scene]
    else:
        raise ValueError(f'Invalid scene: {scene}')


def get_checkpoint_path(args) -> str:
    """Create checkpoint directory path based on arguments."""
    base_dir = './checkpoints' if 'u8' in os.path.expanduser('~') else '~/checkpoints'
    suffix = f"{args.suffix}" if args.suffix else ""
    
    return os.path.expanduser(
        f'{base_dir}/yobo_results/{args.experiment}/{args.scene}_{args.stage}{suffix}'
    )


def get_partial_checkpoint_path(args) -> Optional[str]:
    """Get path for partial checkpoint if specified in arguments."""
    if not args.take_stage:
        return None
        
    base_dir = './checkpoints' if 'u8' in os.path.expanduser('~') else '~/checkpoints'
    return os.path.expanduser(
        f'{base_dir}/yobo_results/{args.experiment}/{args.scene}_{args.take_stage}'
    )


def parse_stage_flags(args) -> Dict[str, Any]:
    """Parse stage string and set appropriate flags."""
    flags = {}
    
    # Handle resample flags
    if 'resample_depth' in args.stage:
        flags['resample'] = True
        flags['resample_render'] = True
        flags['resample_depth'] = True
        args.stage = args.stage.replace('_resample_depth', '')
    elif 'resample' in args.stage:
        flags['resample'] = True
        flags['resample_render'] = True
        args.stage = args.stage.replace('_resample', '')

    # Handle illumination flags
    if 'rotate_illum' in args.stage:
        flags['multi_illum'] = True
        flags['rotate_illum'] = True
        args.stage = args.stage.replace('_rotate_illum', '')
    elif 'multi_illum' in args.stage:
        flags['multi_illum'] = True
        args.stage = args.stage.replace('_multi_illum', '')
        
    return flags


def build_command(args, checkpoint_dir: str, partial_checkpoint_dir: Optional[str]) -> list:
    """Build the command list for training."""
    cmd = [
        'python', '-m', 'train_with_trainer',
        f'--gin_configs=configs/{args.config_file}.gin',
        
        # Trainer bindings
        f'--gin_bindings="Trainer.stage=\'{args.stage}\'"',
        f'--gin_bindings="Trainer.vis_extra={args.vis_extra}"',
        f'--gin_bindings="Trainer.vis_secondary={args.vis_secondary}"',
        f'--gin_bindings="Trainer.vis_light_sampler={args.vis_light_sampler}"',
        f'--gin_bindings="Trainer.vis_surface_light_field={args.vis_surface_light_field}"',
        f'--gin_bindings="Trainer.viewer_only={args.viewer_only}"',
        f'--gin_bindings="Trainer.vis_only={args.vis_only}"',
        f'--gin_bindings="Trainer.vis_restart={args.vis_restart}"',
        f'--gin_bindings="Trainer.vis_start={args.vis_start}"',
        f'--gin_bindings="Trainer.vis_end={args.vis_end}"',
        
        f'--gin_bindings="Trainer.stopgrad={args.stopgrad}"',
        f'--gin_bindings="Trainer.resample={args.resample}"',
        f'--gin_bindings="Trainer.resample_depth={args.resample_depth}"',
        f'--gin_bindings="Trainer.sample_factor={args.sample_factor}"',
        f'--gin_bindings="Trainer.num_resample={args.num_resample}"',
        f'--gin_bindings="Trainer.resample_render={args.resample_render}"',
        f'--gin_bindings="Trainer.sample_render_factor={args.sample_render_factor}"',
        f'--gin_bindings="Trainer.render_repeats={args.render_repeats}"',
        f'--gin_bindings="Trainer.relight={args.relight}"',
        f'--gin_bindings="Trainer.save_results={not args.no_save}"',
        
        # Config bindings
        f'--gin_bindings="Config.checkpoint_dir=\'{checkpoint_dir}\'"',
        f'--gin_bindings="Config.train_render_every=1000"',
        f'--gin_bindings="Config.no_vis={args.no_vis}"',
        f'--gin_bindings="Config.vis_decimate={args.vis_decimate}"',
        
        f'--gin_bindings="Config.train_length_mult={args.train_length_mult}"',
        f'--gin_bindings="Config.lr_factor_mult={args.lr_factor_mult}"',
        f'--gin_bindings="Config.batch_size={args.batch_size}"',
        f'--gin_bindings="Config.render_chunk_size={args.render_chunk_size}"',
        f'--gin_bindings="Config.grad_accum_steps={args.grad_accum_steps}"',
        f'--gin_bindings="Config.secondary_grad_accum_steps={args.secondary_grad_accum_steps}"',
        
        # More config bindings
        f'--gin_bindings="Config.multi_illumination={args.multi_illum}"',
        f'--gin_bindings="Config.rotate_illumination={args.rotate_illum}"',
        f'--gin_bindings="Config.vis_render_path={args.vis_render_path}"',
        f'--gin_bindings="Config.fixed_light={args.fixed_light}"',
        f'--gin_bindings="Config.fixed_camera={args.fixed_camera}"',
        f'--gin_bindings="Config.light_transform_idx={args.light_idx}"',
        f'--gin_bindings="Config.vis_only={args.vis_only}"',
        f'--gin_bindings="Config.compute_relight_metrics={args.relight}"',
        f'--gin_bindings="Config.sl_relight={args.sl_relight}"',
        f'--gin_bindings="Config.filter_median={args.filter_median}"',
        f'--gin_bindings="Config.round_roughness={args.round_roughness}"',
        f'--gin_bindings="Config.eval_train={args.eval_train}"',
        f'--gin_bindings="Config.eval_path={args.eval_path}"',
        
        '--logtostderr'
    ]
    
    # Add conditional bindings
    if args.relight:
        cmd.append(f'--gin_bindings="Config.env_map_name=\'{args.env_map_name}\'"')
        
    if partial_checkpoint_dir:
        cmd.append(f'--gin_bindings="Config.partial_checkpoint_dir=\'{partial_checkpoint_dir}\'"')
        
    if args.early_exit_steps > 0:
        cmd.append(f'--gin_bindings="Config.early_exit_steps={args.early_exit_steps}"')
        
    if args.suffix is not None and ('albedo_least' in args.suffix):
        cmd.append(f'--gin_bindings="Trainer.albedo_correct_median=False"')
        cmd.append(f'--gin_bindings="Trainer.albedo_clip=1.0"')
        
    return cmd


def train_one_stage_partial(args):
    """Main function to set up and execute a training stage."""
    # Set default config file if not specified
    if not args.config_file:
        args.config_file = get_config_file(args.scene)
        
    # Get checkpoint directories
    checkpoint_dir = get_checkpoint_path(args)
    partial_checkpoint_dir = get_partial_checkpoint_path(args)
    
    # Parse stage flags and update args
    stage_flags = parse_stage_flags(args)
    for key, value in stage_flags.items():
        setattr(args, key, value)
    
    # Build and execute command
    cmd = build_command(args, checkpoint_dir, partial_checkpoint_dir)
    cmd_str = ' '.join(cmd)
    print(f'Executing command: {cmd_str}')
    os.system(cmd_str)


def setup_argparser():
    """Set up the argument parser with all needed options."""
    parser = argparse.ArgumentParser(description='Train one stage of the YOBO material model.')
    
    # Basic options
    parser.add_argument('--suffix', help='suffix to add to checkpoint dir')
    parser.add_argument('-s', '--scene', default='roomy', help='the scene to train on (default: roomy)')
    parser.add_argument('-t', '--stage', help='the stage to train')
    parser.add_argument('-p', '--take_stage', help='the stage to take the partial checkpoint from')
    parser.add_argument('-e', '--experiment', default='synthetic', help='the experiment to run (default: synthetic)')
    parser.add_argument('-c', '--config_file', help='the config file to use')
    
    # Visualization options
    parser.add_argument('-v', '--viewer_only', action='store_true', help='whether to use the viewer only')
    parser.add_argument('-l', '--vis_only', action='store_true', help='whether to log only')
    parser.add_argument('--vis_restart', action='store_true', help='restart visualization')
    parser.add_argument('--vis_start', type=int, default=0, help='visualization start frame')
    parser.add_argument('--vis_end', type=int, default=200, help='visualization end frame')
    parser.add_argument('--no_vis', action='store_true', help='disable visualization')
    parser.add_argument('--vis_decimate', type=int, default=1, help='decimation factor for visualization')
    parser.add_argument('--vis_render_path', action='store_true', help='visualize render path')
    parser.add_argument('--vis_extra', action='store_true', help='visualize extra information')
    parser.add_argument('--vis_secondary', action='store_true', help='visualize secondary rays')
    parser.add_argument('--vis_light_sampler', action='store_true', help='visualize light sampler')
    parser.add_argument('--vis_surface_light_field', action='store_true', help='visualize surface light field')
    
    # Rendering and light options
    parser.add_argument('--light_idx', type=int, default=0, help='light transform index')
    parser.add_argument('--relight', action='store_true', help='perform relighting')
    parser.add_argument('--sl_relight', action='store_true', help='perform surface light relighting')
    parser.add_argument('--filter_median', action='store_true', help='apply median filter')
    parser.add_argument('--round_roughness', action='store_true', help='round roughness values')
    parser.add_argument('--no_gaussian', action='store_true', help='disable gaussian filter')
    parser.add_argument('--eval_train', action='store_true', help='evaluate on training data')
    parser.add_argument('--eval_path', action='store_true', help='evaluate on path')
    parser.add_argument('--env_map_name', help='environment map name for relighting')
    parser.add_argument('--fixed_light', action='store_true', help='use fixed light position')
    parser.add_argument('--fixed_camera', action='store_true', help='use fixed camera position')
    
    # Sampling options
    parser.add_argument('--resample_render', action='store_true', help='enable render resampling')
    parser.add_argument('--sample_render_factor', type=int, default=2, help='render sample factor')
    parser.add_argument('--render_repeats', type=int, default=1, help='number of render repeats')
    parser.add_argument('--resample', action='store_true', help='enable resampling')
    parser.add_argument('--resample_depth', action='store_true', help='enable depth resampling')
    parser.add_argument('--num_resample', type=int, default=1, help='number of resamples')
    parser.add_argument('--sample_factor', type=int, default=2, help='sample factor')
    
    # Model options
    parser.add_argument('--stopgrad', action='store_true', help='stop gradient for material geometry')
    parser.add_argument('--multi_illum', action='store_true', help='use multiple illumination')
    parser.add_argument('--rotate_illum', action='store_true', help='use rotated illumination')
    
    # Training options
    parser.add_argument('--train_length_mult', type=int, default=1, help='training length multiplier')
    parser.add_argument('--lr_factor_mult', type=float, default=1.0, help='learning rate factor multiplier')
    parser.add_argument('--batch_size', type=int, default=16384, help='batch size')
    parser.add_argument('--render_chunk_size', type=int, default=4096, help='render chunk size')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--secondary_grad_accum_steps', type=int, default=1, help='secondary gradient accumulation steps')
    parser.add_argument('--early_exit_steps', type=int, default=-1, help='steps before early exit')
    parser.add_argument('--no_save', action='store_true', help='disable saving results')

    return parser


if __name__ == '__main__':
    parser = setup_argparser()
    args = parser.parse_args()
    train_one_stage_partial(args)