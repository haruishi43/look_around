# SUN360 Dataset for FindView

## How the datasets are divided

By scenes:
- Main scenes: indoor, outdoor, other
- Sub scenes: only for indoor and outdoor

By difficulties (Abstract; the details are in the code):
- easy (target is near initial rot; can see in FOV)
- medium (a little bit far; can see a little in FOV)
- hard (farther; can't see in FOV)

NOTE: I think `difficulties` are used in scheduling to make it so that the agent can gradually
understand the task

NOTE: Prior criteria: ASSUMPTIONS!
- Need to make sure that the agent can reach the target
- Set yaw and pitch movements to increments of 1 degree (integers!)
- Using integers saves space too
- We also need to set the threshold of the upper and lower bound to pitch direction
- Since the FOV is usually 90 degrees, if you tilt 45 degrees, you can see the pole
- Maybe set the threshold to 60?
- Pitch should be normal distribution


## Splits for the dataset

We divide the dataset (SUN360) into 3 sets:
- Train
- Validation
- Test

Notes:
- Validation (Val), Test data should be static
- Val is used to check if the agents are not overfitting during training
- Training data can be randomly generated

## Episodes

A single `Episode` includes the following:
- episode_id
- img_name
- path
- label (scene)
- sub_label (sub scene)
- initial_rotation
- target_rotation
- difficulty
- steps_for_shortest_path (hints for training)

A single `PseudoEpisode` includes the following:
- img_name
- path
- label (scene)
- sub_label (sub scene)
