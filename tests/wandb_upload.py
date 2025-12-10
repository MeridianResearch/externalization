import os
import wandb

PROJECT = "early-exit-RL-test"
ENTITY = "vkarthik095-university-of-amsterdam"
RUN_ID = "ctv5haer" #change to whatever run you need
CKPT_DIR = "models/rl_20251204_tom_rlmodel_batch4_k6_lambda0.1" #change folder
ARTIFACT_NAME = "model-checkpoints-lambda-0_1"

run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    id=RUN_ID,
    resume="allow",
    job_type="upload-checkpoints",
)

artifact = wandb.Artifact(
    name=ARTIFACT_NAME,
    type="model",
    description="Checkpoints from run",
    metadata={"num_checkpoints": len(os.listdir(CKPT_DIR))},
)

artifact.add_dir(CKPT_DIR)

run.log_artifact(artifact)

run.finish()
