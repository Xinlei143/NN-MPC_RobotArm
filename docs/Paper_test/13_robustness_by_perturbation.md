# Robustness by Perturbation

ThreadedASAP minus Projected Direct IK TCP RMSE was negative in every condition:
nominal -10.84 mm, payload -23.33 mm, actuator gain -11.13 mm, force pulse
-10.14 mm, and observation noise -5.68 mm. All corresponding confidence
intervals excluded zero.

Against Preview IK, the observation-noise estimate was +0.95 mm with a CI that
crossed zero. Threaded versus FullVirtual also showed a measurable +3.20 mm
loss under actuator-gain disturbance. The robustness claim is therefore
conditioned: threaded execution improves on Projected IK throughout this test
matrix, but it does not dominate Preview IK under every disturbance.
