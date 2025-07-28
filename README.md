This repository contains several versions of nuclearcraft. The "with more relaxation" is the newest version.

You need to get Julia Gurobi working. Follow instructions here. 
https://jump.dev/JuMP.jl/stable/installation/

The function prints out the reactor design that maximizes the power generation given constraints. The function argument names are self-documenting.

function nuclearcraftoptimize_relaxed(base_energy, base_heat,reactor_width, reactor_length, reactor_height, reactor_cell_limit = 20000;
water_cooling = 60, water_limit = 20000, redstone_cooling = 90, redstone_limit = 20000,
quartz_cooling = 90, quartz_limit = 20000, gold_cooling = 120, gold_limit = 20000, glowstone_cooling = 130, glowstone_limit = 20000,
lapis_cooling=120,lapis_limit=20000,diamond_cooling=150,diamond_limit=20000,liquid_helium_cooling = 140, liquid_helium_limit = 20000,
enderium_cooling = 120, enderium_limit = 20000, cryotheum_cooling = 160, cryotheum_limit = 20000,
iron_cooling = 80, iron_limit=20000, emerald_cooling = 160,emerald_limit=20000, copper_cooling=80, copper_limit=20000,
tin_cooling = 120, tin_limit=20000, magnesium_cooling = 110, magnesium_limit = 20000, num_threads = 1, time_limit = 0.0
)