from rxrx1.utils.visualization import plot_experiments

seed_comparision_experiment_names = ['resnet18_seed0_size256',
                                     'resnet18_seed42_size256',
                                     'resnet18_seed2026_size256',
                                     'resnet18_seed2386_size256',
                                     'resnet18_seed3407_size256']

image_size_comparision_experiment_names = ['resnet18_size128_seed42',
                                           'resnet18_seed42_size256',
                                           'resnet18_size384_seed42',
                                           'resnet18_size512_seed42']

plot_experiments(seed_comparision_experiment_names, 'seed_comparision',overwrite=False,
                 linewidth=1.5, alpha=0.7)
plot_experiments(image_size_comparision_experiment_names, 'image_size_comparision',overwrite=False)