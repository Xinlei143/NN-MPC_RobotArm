# Model-A Replica Training

Replica models support the optional uncertainty supervisor. Train each replica
on the same compatible dataset and normalizer as the primary Model-A model, but
with a distinct random seed. Replicas monitor the selected CEM trajectory; they
do not participate in CEM optimization or average their predictions into the
control action.
