from pictor.xomics import Omix
import numpy as np



# -----------------------------------------------------------------------------
# (1) Prepare data
# -----------------------------------------------------------------------------
n_samples = 100
n_features = 200

features = np.random.randn(n_samples, n_features)
targets = np.random.randint(0, 2, n_samples)
feature_labels = [f'Feature-{i + 1}' for i in range(n_features)]
sample_labels = [f'Sample-{i + 1}' for i in range(n_samples)]
target_labels = ['Class-0', 'Class-1']

# -----------------------------------------------------------------------------
# (2) Wrap data
# -----------------------------------------------------------------------------
omix = Omix(
  features=features,
  targets=targets,
  feature_labels=feature_labels,
  sample_labels=sample_labels,
  target_labels=target_labels,
  data_name='Demo'
)


