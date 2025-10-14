# TODOs

## Code
1. Improve data loading
2. Models
3. Testing

## Comments
1. Mae dataset + datamodule: violates separation of concerns:
   - reading of .npy file + header happens in dataset
   - random cropping happens in dataset with a volatile random state that cannot be controlled externally. What happens if training crashes? How do you restore random state? Experiments are not truly reproducible (different cropping) if the global rng changes, which happens when you compare frameworks and add something that consumes the global rng in advance.   
   - rest happens in transforms, which are externally defined, usually through MONAI. MONAI allows external control of the internal state per transform, contrary to the cropping mechanism.
   - masking --> happens in pl_module, again random state not controlled. Also, masking ratio is fixed, while it has been recommended by state of the art approaches to vary it randomly, between 0.6 and 0.75.
   - mixed traversal; participant-level in the datamodule, rest in dataset

   So, MONAI can handle almost every of the above operations, except path population: loading the numpy (the header is not even needed, since we know it's RAS with 1mm3 isotropic spacing) + cropping + masking pattern + augmentations: all random states controlled by Compose().set_random_state().

2. Contrastive dataset + datamodule: same, plus: 
   - shared cropping: As discussed, I believe that truly meaningful comparisons should ensure that the same conditions are applied to both positive and negative pairs. As is, this biases the alignment with positive pairs. But let's say we keep it as is, I would recommend to increase patch size to 128, if your GPU allows it as well. 
   - each combination is treated as a new input. Would this not lead to a massive number of steps per epoch?
   - Shouldn't we shuffle the pairs, so that there is more cross-subject variability per batch? 

I recommend modifying only the transforms for this project. But in the future this needs to be looked up. 
   
3. Models:
   - Contrastive: Queue size: small (4096), can do larger one (16384) given that contrastive is taking all combinations of modalities in each epoch. 
   - Contrastive: Do you need to compute the loss for view1-view2 and then view2-view1, given that you generate all modality combinations (I assume this distinguishes 1-2 from 2-1) as training points? 
   - Combined: Since contrastive loss is not truly informative in the first epochs, I would recommend that training is dominated by the MAE loss. It should be scheduled, and gradually increased to 50-50. 

4. Testing:
   - Ideally mock patient directory structure, since I don't (and hence future users) have access to the test dir. 

## Changes
1. Moved cropping to transforms as well, since it is good if it happens after rotation augmentations to minimize artifacts.
2. Should I move loading as well?