# -------------
# detection API summary
# PROMPT
# ['battery','slot'] 

battery outside the crate

# RESPONSE
battery_list = self.get_obj('battery')
tgt_idx = self.find_instance_in_category(instance = 'battery outside the crate', category = 'battery')
tgt_battery_list = []
for idx in tgt_idx:
    tgt_battery_list.append(battery_list[idx]) 
output_var = tgt_battery_list
# -------------
# -------------
# find_instance_in_category API summary
# PROMPT
# The first prompt is #battery outside the crate

# RESPONSE
[0, 1]
# -------------
# -------------
# detection API summary
# PROMPT
# ['battery','slot'] 

slot

# RESPONSE
slot_list = self.get_obj('slot')
output_var = slot_list
# -------------
