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

# Stats

## Indoor

old_building 	 230
dining_room 	 56
childs_room_daycare 	 40
bedroom 	 423
subway_station 	 133
elevator 	 1
theater 	 172
massage_room 	 2
conference_room 	 41
kitchen 	 53
train_interior 	 184
aquarium 	 2
library 	 26
museum 	 562
hangar 	 22
classroom 	 46
cave 	 113
stadium 	 48
corridor 	 336
belfry 	 5
pilothouse 	 11
tent 	 25
studio 	 20
lobby_atrium 	 685
airplane_interior 	 15
laboratory 	 4
others 	 7430
mechanical_room 	 18
jail_cell 	 3
legislative_chamber 	 12
church 	 1270
stable 	 12
hospital_room 	 31
expo_showroom 	 302
closet 	 3
indoor_pool 	 48
staircase 	 66
living_room 	 268
restaurant 	 948
gym 	 28
car_interior 	 34
bathroom 	 125
greenhouse 	 21
office 	 60
sauna 	 2
shop 	 320
tomb 	 2
observatory 	 6
workshop 	 80
parking_garage 	 14
50 of 50
14358

## Outdoor

train_station_or_track 	 40
lawn 	 100
field 	 330
plaza_courtyard 	 690
underwater 	 4
mountain 	 257
garden 	 2
gulch 	 3
others 	 49461
swimming_pool 	 36
sports_field 	 11
jetty 	 37
bridge 	 64
boat_deck 	 73
beach 	 196
cemetery 	 37
balcony 	 67
highway 	 13
patio 	 17
ruin 	 112
coast 	 78
park 	 160
arena 	 20
forest 	 239
skatepark 	 1
street 	 646
airport 	 9
wharf 	 111
construction_site 	 8
parking_lot 	 66
amphitheatre 	 3
desert 	 47
32 of 32
52938
