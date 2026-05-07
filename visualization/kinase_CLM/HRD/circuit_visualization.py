from pymol import cmd

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_trace_mode', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)

# Custom Colors
cmd.set_color('base_blue', [91/255, 150/255, 210/255])

# Load Structure
cmd.load('P83104_5_281_4o91.1.A.cif', 'base_struct')
cmd.hide('everything', 'base_struct')
cmd.color('base_blue', 'base_struct')
cmd.show('cartoon', 'base_struct')

# Coloring Helper (Normalized)
def apply_spectrum_norm(obj_name, raw_max):
    cmd.color('base_blue', obj_name)
    print(f'Object {obj_name}: Raw Max = {raw_max:.4f}')
    if raw_max < 0.0001: return
    selection = f'{obj_name} and b > 0.1'
    cmd.spectrum('b', 'white_red', selection=selection, minimum=0.1, maximum=1.0)

# --- L8_897 (Raw Max: 6.6246) ---
cmd.create('L8_897', 'base_struct')
cmd.alter('L8_897', 'b=0.0')
cmd.alter('L8_897 and chain A and resi 128', 'b=0.1817')
cmd.alter('L8_897 and chain A and resi 131', 'b=0.2381')
cmd.alter('L8_897 and chain A and resi 132', 'b=1.0000')
apply_spectrum_norm('L8_897', 6.6246490478515625)
cmd.group('Circuit_Analysis', 'L8_897')

# --- L5_1090 (Raw Max: 15.9321) ---
cmd.create('L5_1090', 'base_struct')
cmd.alter('L5_1090', 'b=0.0')
cmd.alter('L5_1090 and chain A and resi 124', 'b=0.5659')
cmd.alter('L5_1090 and chain A and resi 125', 'b=0.3645')
cmd.alter('L5_1090 and chain A and resi 126', 'b=0.4759')
cmd.alter('L5_1090 and chain A and resi 127', 'b=0.7698')
cmd.alter('L5_1090 and chain A and resi 128', 'b=0.2576')
cmd.alter('L5_1090 and chain A and resi 129', 'b=0.6161')
cmd.alter('L5_1090 and chain A and resi 130', 'b=0.7301')
cmd.alter('L5_1090 and chain A and resi 131', 'b=1.0000')
cmd.alter('L5_1090 and chain A and resi 132', 'b=0.8717')
apply_spectrum_norm('L5_1090', 15.93206787109375)
cmd.group('Circuit_Analysis', 'L5_1090')

# --- L1_3183 (Raw Max: 4.7164) ---
cmd.create('L1_3183', 'base_struct')
cmd.alter('L1_3183', 'b=0.0')
cmd.alter('L1_3183 and chain A and resi 26', 'b=0.3996')
cmd.alter('L1_3183 and chain A and resi 53', 'b=0.7063')
cmd.alter('L1_3183 and chain A and resi 68', 'b=0.6673')
cmd.alter('L1_3183 and chain A and resi 72', 'b=0.4728')
cmd.alter('L1_3183 and chain A and resi 110', 'b=0.9278')
cmd.alter('L1_3183 and chain A and resi 127', 'b=1.0000')
cmd.alter('L1_3183 and chain A and resi 132', 'b=0.9721')
apply_spectrum_norm('L1_3183', 4.716363906860352)
cmd.group('Circuit_Analysis', 'L1_3183')

# --- L2_1754 (Raw Max: 1.5147) ---
cmd.create('L2_1754', 'base_struct')
cmd.alter('L2_1754', 'b=0.0')
cmd.alter('L2_1754 and chain A and resi 26', 'b=0.4422')
cmd.alter('L2_1754 and chain A and resi 68', 'b=0.7999')
cmd.alter('L2_1754 and chain A and resi 72', 'b=0.4195')
cmd.alter('L2_1754 and chain A and resi 110', 'b=0.0688')
cmd.alter('L2_1754 and chain A and resi 127', 'b=0.6541')
cmd.alter('L2_1754 and chain A and resi 132', 'b=1.0000')
apply_spectrum_norm('L2_1754', 1.5146578550338745)
cmd.group('Circuit_Analysis', 'L2_1754')

# --- L7_2070 (Raw Max: 6.4646) ---
cmd.create('L7_2070', 'base_struct')
cmd.alter('L7_2070', 'b=0.0')
cmd.alter('L7_2070 and chain A and resi 67', 'b=0.1949')
cmd.alter('L7_2070 and chain A and resi 69', 'b=0.9324')
cmd.alter('L7_2070 and chain A and resi 70', 'b=0.3211')
cmd.alter('L7_2070 and chain A and resi 72', 'b=0.6611')
cmd.alter('L7_2070 and chain A and resi 73', 'b=0.1931')
cmd.alter('L7_2070 and chain A and resi 90', 'b=0.6391')
cmd.alter('L7_2070 and chain A and resi 91', 'b=1.0000')
cmd.alter('L7_2070 and chain A and resi 94', 'b=0.6290')
cmd.alter('L7_2070 and chain A and resi 98', 'b=0.3171')
cmd.alter('L7_2070 and chain A and resi 99', 'b=0.3644')
cmd.alter('L7_2070 and chain A and resi 127', 'b=0.5639')
cmd.alter('L7_2070 and chain A and resi 129', 'b=0.9984')
cmd.alter('L7_2070 and chain A and resi 130', 'b=0.9070')
cmd.alter('L7_2070 and chain A and resi 131', 'b=0.5279')
cmd.alter('L7_2070 and chain A and resi 132', 'b=0.7170')
apply_spectrum_norm('L7_2070', 6.4645562171936035)
cmd.group('Circuit_Analysis', 'L7_2070')

cmd.disable('base_struct')
cmd.disable('Circuit_Analysis')
cmd.zoom('base_struct')
print('Done! Enable specific objects in Circuit_Analysis to view.')