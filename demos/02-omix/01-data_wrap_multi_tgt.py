from pictor.xomics import Omix
import numpy as np



# -----------------------------------------------------------------------------
# (1) Prepare data
# -----------------------------------------------------------------------------
n_samples = 100
n_features = 200

features = np.random.randn(n_samples, n_features)
feature_labels = [f'Feature-{i + 1}' for i in range(n_features)]
sample_labels = [f'Sample-{i + 1}' for i in range(n_samples)]

target_1_key = 'Grades'
targets_1 = np.random.randint(0, 2, n_samples)
target_labels_1 = ['Grade-0', 'Grade-1']

target_2_key = 'Risks'
targets_2 = np.random.randint(0, 2, n_samples)
target_labels_2 = ['Low Risk', 'High Risk']

# -----------------------------------------------------------------------------
# (2) Wrap data
# -----------------------------------------------------------------------------
omix = Omix(
  features=features,
  targets=targets_1,
  feature_labels=feature_labels,
  sample_labels=sample_labels,
  target_labels=target_labels_1,
  data_name='Demo'
)

omix.add_to_target_collection(target_1_key, targets_1, target_labels_1)
omix.add_to_target_collection(target_2_key, targets_2, target_labels_2)

# Switch to target-2
omix.set_targets(target_2_key, return_new_omix=False)

