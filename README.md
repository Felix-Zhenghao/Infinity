# What other things should be done to finetune a vae

- In the `Infinity.infinity.models.bsq_vae.vae.py`
```python

##################
# Use this if load finetuned checkpoint
##################
new_state_dict = {}
for key in state_dict:
    if key.startswith("vae."):
        new_state_dict[key[4:]] = state_dict[key]
    else:
        new_state_dict[key] = state_dict[key]
vae.load_state_dict(new_state_dict)

##################
# Use this if load pretrained checkpoint (by Infinity team)
##################
if state_dict:
    if args.ema == "yes":
        vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["ema"], prefix="", expand=False)
    if "vae" not in state_dict.keys():
        vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict, prefix="", expand=False)
    else:
        vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["vae"], prefix="", expand=False)

```