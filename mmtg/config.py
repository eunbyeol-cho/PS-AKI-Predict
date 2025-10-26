from sacred import Experiment
import os
import json

ex = Experiment("METER", save_git_info=False)

def load_hospital_config(study):
    """Load hospital-specific configuration from JSON file."""
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
    config_path = os.path.join(configs_dir, f'{study}.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found for {study}. Expected at: {config_path}\n"
            f"Please run preprocess.py first to generate the config file."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Register hospital config loader as a captured config
@ex.config_hook
def add_hospital_config(config, command_name, logger):
    """Hook to add hospital-specific config after named configs are loaded."""
    study = config.get('study')
    
    # If study is provided, load hospital-specific configs
    if study:
        try:
            hospital_config = load_hospital_config(study)
            
            # Calculate cat_vocab_size: cat_null_id + 3 (null + pad + mask)
            cat_null_id = hospital_config['cat_null_ids']
            cat_vocab_size = cat_null_id + 3
            
            # Override config with hospital-specific values
            config.update({
                'nc': hospital_config['cat_col_num'],
                'nn': hospital_config['num_col_num'],
                'ntext': hospital_config.get('text_num'),
                'num_null_id': hospital_config['num_null_ids'],
                'cat_null_id': cat_null_id,
                'cat_vocab_size': cat_vocab_size,
                'column_order': hospital_config['column_order'],
                'cat_columns_numclass_list': hospital_config['cat_columns_numclass_list']
            })
        except FileNotFoundError:
            logger.warning(f"Could not load config for {study}. Using defaults.")
    
    return config

@ex.config
def config():
    ###################### Make your own configs ######################
    # Paths - will be auto-configured based on study parameter
    input_path = None  # Set automatically based on study
    output_path = None  # Set automatically based on study
    norm = 'znorm'
    test_sets = ['test']

    # These will be automatically loaded from hospital-specific JSON configs
    nc = None  # number of categorical columns
    nn = None  # number of numerical columns
    ntext = None  # number of text tokens
    num_null_id = None
    cat_null_id = None
    cat_vocab_size = None  # Will be cat_null_id + 3 (null + pad + mask)
    column_order = None
    cat_columns_numclass_list = None
    
    #######################################################################

    tokenizer = 'bert-base-multilingual-cased'
    text_vocab_size = 119547 #bert-base-multilingual-cased

    #Model settings
    modeling = 'MLM'
    sep_regressor = False
    sep_numemb = False
    bert_type = "mini"
    var_head_type = 'abs'

    #Training settings
    n_epochs = 500
    patience = 30
    dropout = 0.1

    #Below params varies with the environment
    modality = ['table']
    mask_scheduler = 'pow1'
    test_mlm_probability = 1.0
    lr = 5e-5
    seed = 2020
    num_loss_weight = 0.1

    num_nodes = 1 
    num_gpus = 1
    per_gpu_batchsize = 64

    test_only = False
    resume = False
    debug = False
    
    save_dir = 'checkpoints'
    wandb_project_name = 'MMTG'

    re_sampling = False
    iter_decoding = False
    n_iter = 10

    temperature = 1.0
    topk_filter_thres = 0.9
    sigmoid_temperature = 1
    unmask_by = ""
    sample_by = ""
    null_sample = True

    study = None
    boxcox_transformation = True
    target_study=None
    fixed_test_mask = True
    spark_features = ["A_Sex", "Op_Dep", "Op_Type", "AB_DM", "D_RASB_90", "A_Age", "B_Cr_near", "Op_EST_Dur"]
    
@ex.named_config
def task_train_both2table():
    modality = ['table', 'text']


@ex.named_config
def task_train_table2table():
    modality = ['table']
    # Hospital-specific config will be loaded dynamically from configs/{study}.json


@ex.named_config
def task_test_table2table():
    modality = ['table']
    test_sets=["train", "valid"]
    debug = True
    test_only = True
    iter_decoding = True
    n_iter = 10
    sample_by = "categorical"
    unmask_by = "random"
    per_gpu_batchsize = 2048
    # Hospital-specific config will be loaded dynamically from configs/{study}.json

@ex.named_config
def task_test_both2table():
    modality = ['table', 'text']
    test_sets=["train", "valid"]
    
    debug = True
    test_only = True
    iter_decoding = True
    n_iter = 10
    sample_by = "categorical"
    unmask_by = "random"
    per_gpu_batchsize = 2048


# All hospital-specific configs have been moved to configs/{study}.json files
# Use task_train_table2table or task_test_table2table with study parameter
# Example: python main.py with task_train_table2table study=SNUH
