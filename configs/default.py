
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Optimizer.
  config.optimizer = 'adam'
  config.learning_rate = 1e-3

  # Training hyperparameters.
  config.pre_train = True
  config.batch_size = 256
  config.max_pad = 2**10
  config.num_train_steps = 6_000
  config.warmup_steps = 100
  config.log_every_steps = 10
  config.eval_every_steps = 2_000
  config.checkpoint_every_steps = 3_000
  config.add_virtual_node = False
  config.add_undirected_edges = True
  config.add_self_loops = True
  config.half_precision = True

  # GNN hyperparameters.
  config.model = 'GraphConvNet'
  config.message_passing_steps = 5
  config.latent_size = 256
  config.dropout_rate = 0.1
  config.num_mlp_layers = 2
  config.num_para = 7
  config.skip_connections = True
  config.layer_norm = True
  return config
