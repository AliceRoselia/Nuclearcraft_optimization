using JuMP
using Gurobi


function get_neighbors(x,y,z,reactor_width,reactor_length,reactor_height)
    result = Vector{NTuple{3,Int}}()
    if x > 1
        push!(result,(x-1,y,z))
    end
    if y > 1
        push!(result,(x,y-1,z))
    end
    if z > 1
        push!(result,(x,y,z-1))
    end
    
    if x < reactor_width
        push!(result,(x+1,y,z))
    end
    if y < reactor_length
        push!(result,(x,y+1,z))
    end
    if z < reactor_height
        push!(result,(x,y,z+1))
    end

    return result
end


function nuclearcraftoptimize(base_energy, base_heat,reactor_width, reactor_length, reactor_height, reactor_cell_limit = 20000;
    water_cooling = 60, water_limit = 20000, redstone_cooling = 90, redstone_limit = 20000,
    quartz_cooling = 90, quartz_limit = 20000, gold_cooling = 120, gold_limit = 20000, glowstone_cooling = 130, glowstone_limit = 20000,
    lapis_cooling=120,lapis_limit=20000,diamond_cooling=150,diamond_limit=20000,liquid_helium_cooling = 140, liquid_helium_limit = 20000,
    enderium_cooling = 120, enderium_limit = 20000, cryotheum_cooling = 160, cryotheum_limit = 20000,
    iron_cooling = 80, iron_limit=20000, emerald_cooling = 160,emerald_limit=20000, copper_cooling=80, copper_limit=20000,
    tin_cooling = 120, tin_limit=20000, magnesium_cooling = 110, magnesium_limit = 20000, num_threads = 1, time_limit = 0.0
    )
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "Threads", num_threads)
    if time_limit !== 0.0
        set_time_limit_sec(model,time_limit)
    end
    set_string_names_on_creation(model,false)
    set_attribute(model,"MIPFocus",2)
    set_attribute(model,"FlowCoverCuts",2) #Max value for flow cover.
    set_attribute(model,"MIRCuts",2)
    set_attribute(model,"RelaxLiftCuts",2)
    set_attribute(model,"RLTCuts",2)
    set_attribute(model,"ZeroHalfCuts",2)
    set_attribute(model,"MixingCuts",2)
    set_attribute(model,"CoverCuts",2)
    set_attribute(model,"ImpliedCuts",2)
    set_attribute(model,"ProjImpliedCuts",2)
    set_attribute(model,"BQPCuts",2)
    set_attribute(model,"Method",2) #Interior point method. Faster for large sparse problems such as this one.
    set_attribute(model,"BarConvTol",0.001)
    set_attribute(model,"OptimalityTol",0.01) #Any movement after this tolerance is a lie.
    set_attribute(model,"MIPGapAbs",0.1) #It is provable that if the gap is less than 1/6, then it is optimal. Set to 0.1 for leeway.
    set_attribute(model,"Symmetry",2)
    #set_attribute(model,"mip_heuristic_effort",0.05)
    @variable(model,reactor_cells[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(reactor_cells)<=reactor_cell_limit)
    @variable(model,moderators[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    

    #You can see a weird constraint for x_ward_neutron flux. This is based on cases.
    #Case 1... it is a moderator, in this case, this is at most upward_neutron_flux of below minus 1.
    #Case 2... it is a reactor cell, in this case, it can be set to 4.
    #Case 3... none of the above. In which case, it is zero.
    #Upward boosted.

    #Neutron fluxes are implied ints.
    #x_ward boosted are also implied ints by game theory. (This is due to the tight constraint.)
    @variable(model,upward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,upward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin) #Actually can be unbounded below but for simplicity, let's bound the values.
    @constraint(model,upward_neutron_flux[1:reactor_width,1:reactor_length,1] .<= reactor_cells[1:reactor_width,1:reactor_length,1])
    @constraint(model,upward_neutron_flux[1:reactor_width,1:reactor_length,2:reactor_height] .<= (moderators[1:reactor_width,1:reactor_length,2:reactor_height].+reactor_cells[1:reactor_width,1:reactor_length,2:reactor_height]))
    @constraint(model,upward_neutron_flux[1:reactor_width,1:reactor_length,2:reactor_height] .- (1 .-moderators[1:reactor_width,1:reactor_length,2:reactor_height]) .<= upward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height-1])
    @constraint(model,upward_neutron_flux[1:reactor_width,1:reactor_length,6:reactor_height] .<= 4 .+ reactor_cells[1:reactor_width,1:reactor_length,6:reactor_height] 
    .- moderators[1:reactor_width,1:reactor_length,5:reactor_height-1] .- moderators[1:reactor_width,1:reactor_length,4:reactor_height-2] 
    .- moderators[1:reactor_width,1:reactor_length,3:reactor_height-3] .- moderators[1:reactor_width,1:reactor_length,2:reactor_height-4])
    @constraint(model,upward_boosted_reactor[1:reactor_width,1:reactor_length,1] .== 0) #These dummy variables would likely be optimized away. If not, they stay dummy variables without significant impact.
    @constraint(model,upward_boosted_reactor[1:reactor_width,1:reactor_length,2:reactor_height] .<= upward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height-1])
    @constraint(model,upward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[:,:,1:reactor_height-1] .+ reactor_cells[:,:,2:reactor_height] .- 1 .<= upward_boosted_reactor[:,:,2:reactor_height])
    @constraint(model,reactor_cells[:,:,1:reactor_height-2] .+ moderators[:,:,2:reactor_height-1] .+ reactor_cells[:,:,3:reactor_height] .- 2 .<= upward_boosted_reactor[:,:,3:reactor_height])
    @constraint(model,reactor_cells[:,:,1:reactor_height-3] .+ moderators[:,:,2:reactor_height-2] .+ moderators[:,:,3:reactor_height-1] .+ reactor_cells[:,:,4:reactor_height] .- 3 .<= upward_boosted_reactor[:,:,4:reactor_height])
    @constraint(model,reactor_cells[:,:,1:reactor_height-4] .+ moderators[:,:,2:reactor_height-3] .+ moderators[:,:,3:reactor_height-2] .+ moderators[:,:,4:reactor_height-1] .+ reactor_cells[:,:,5:reactor_height] .- 4 .<= upward_boosted_reactor[:,:,5:reactor_height])
    @constraint(model,reactor_cells[:,:,1:reactor_height-5] .+ moderators[:,:,2:reactor_height-4] .+ moderators[:,:,3:reactor_height-3] .+ moderators[:,:,4:reactor_height-2] .+ moderators[:,:,5:reactor_height-1] .+ reactor_cells[:,:,6:reactor_height] .- 5 .<= upward_boosted_reactor[:,:,6:reactor_height])
    #Downward boosted.
    @variable(model,downward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,downward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,downward_neutron_flux[1:reactor_width,1:reactor_length,reactor_height] .<= reactor_cells[1:reactor_width,1:reactor_length,reactor_height])
    @constraint(model,downward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height-1] .<= (reactor_cells[1:reactor_width,1:reactor_length,1:reactor_height-1] .+ moderators[1:reactor_width,1:reactor_length,1:reactor_height-1]))
    @constraint(model,downward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height-1] .- (1 .-moderators[1:reactor_width,1:reactor_length,1:reactor_height-1]) .<= downward_neutron_flux[1:reactor_width,1:reactor_length,2:reactor_height])
    @constraint(model,downward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height-6] .<= 4 .+ reactor_cells[1:reactor_width,1:reactor_length,1:reactor_height-6] 
    .- moderators[1:reactor_width,1:reactor_length,2:reactor_height-5] .- moderators[1:reactor_width,1:reactor_length,3:reactor_height-4] 
    .- moderators[1:reactor_width,1:reactor_length,4:reactor_height-3] .- moderators[1:reactor_width,1:reactor_length,5:reactor_height-2])
    @constraint(model,downward_boosted_reactor[1:reactor_width,1:reactor_length,reactor_height] .== 0)
    @constraint(model,downward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height-1] .<= downward_neutron_flux[1:reactor_width,1:reactor_length,2:reactor_height])
    @constraint(model,downward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[:,:,2:end] .+ reactor_cells[:,:,1:end-1] .- 1 .<= downward_boosted_reactor[:,:,1:end-1])
    @constraint(model,reactor_cells[:,:,3:end] .+ moderators[:,:,2:end-1] .+ reactor_cells[:,:,1:end-2] .- 2 .<= downward_boosted_reactor[:,:,1:end-2])
    @constraint(model,reactor_cells[:,:,4:end] .+ moderators[:,:,3:end-1] .+ moderators[:,:,2:end-2] .+ reactor_cells[:,:,1:end-3] .- 3 .<= downward_boosted_reactor[:,:,1:end-3])
    @constraint(model,reactor_cells[:,:,5:end] .+ moderators[:,:,4:end-1] .+ moderators[:,:,3:end-2] .+ moderators[:,:,2:end-3] .+ reactor_cells[:,:,1:end-4] .- 4 .<= downward_boosted_reactor[:,:,1:end-4])
    @constraint(model,reactor_cells[:,:,6:end] .+ moderators[:,:,5:end-1] .+ moderators[:,:,4:end-2] .+ moderators[:,:,3:end-3] .+ moderators[:,:,2:end-4] .+ reactor_cells[:,:,1:end-5] .- 5 .<= downward_boosted_reactor[:,:,1:end-5])
    
    #Forward.
    @variable(model,forward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,forward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin)

    
    @constraint(model,forward_neutron_flux[1:reactor_width,1,1:reactor_height] .<= reactor_cells[1:reactor_width,1,1:reactor_height])
    @constraint(model,forward_neutron_flux[1:reactor_width,2:reactor_length,1:reactor_height] .<= (moderators[1:reactor_width,2:reactor_length,1:reactor_height].+reactor_cells[1:reactor_width,2:reactor_length,1:reactor_height]))
    @constraint(model,forward_neutron_flux[1:reactor_width,2:reactor_length,1:reactor_height] .- (1 .-moderators[1:reactor_width,2:reactor_length,1:reactor_height]) .<= forward_neutron_flux[1:reactor_width,1:reactor_length-1,1:reactor_height])
    @constraint(model,forward_neutron_flux[1:reactor_width,6:reactor_length,1:reactor_height] .<= 4 .+ reactor_cells[1:reactor_width,6:reactor_length,1:reactor_height] 
    .- moderators[1:reactor_width,5:reactor_length-1,1:reactor_height] .- moderators[1:reactor_width,4:reactor_length-2,1:reactor_height] 
    .- moderators[1:reactor_width,3:reactor_length-3,1:reactor_height] .- moderators[1:reactor_width,2:reactor_length-4,1:reactor_height])
    @constraint(model,forward_boosted_reactor[1:reactor_width,1,1:reactor_height] .== 0) #These dummy variables would likely be optimized away. If not, they stay dummy variables without significant impact.
    @constraint(model,forward_boosted_reactor[1:reactor_width,2:reactor_length,1:reactor_height] .<= forward_neutron_flux[1:reactor_width,1:reactor_length-1,1:reactor_height])
    @constraint(model,forward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[:,2:end,:] .+ reactor_cells[:,1:end-1,:] .- 1 .<= forward_boosted_reactor[:,2:end,:])
    @constraint(model,reactor_cells[:,3:end,:] .+ moderators[:,2:end-1,:] .+ reactor_cells[:,1:end-2,:] .- 2 .<= forward_boosted_reactor[:,3:end,:])
    @constraint(model,reactor_cells[:,4:end,:] .+ moderators[:,3:end-1,:] .+ moderators[:,2:end-2,:] .+ reactor_cells[:,1:end-3,:] .- 3 .<= forward_boosted_reactor[:,4:end,:])
    @constraint(model,reactor_cells[:,5:end,:] .+ moderators[:,4:end-1,:] .+ moderators[:,3:end-2,:] .+ moderators[:,2:end-3,:] .+ reactor_cells[:,1:end-4,:] .- 4 .<= forward_boosted_reactor[:,5:end,:])
    @constraint(model,reactor_cells[:,6:end,:] .+ moderators[:,5:end-1,:] .+ moderators[:,4:end-2,:] .+ moderators[:,3:end-3,:] .+ moderators[:,2:end-4,:] .+ reactor_cells[:,1:end-5,:] .- 5 .<= forward_boosted_reactor[:,6:end,:])
    
    #Backward
    @variable(model,backward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,backward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,backward_neutron_flux[1:reactor_width,reactor_length,1:reactor_height] .<= reactor_cells[1:reactor_width,reactor_length,1:reactor_height])
    @constraint(model,backward_neutron_flux[1:reactor_width,1:reactor_length-1,1:reactor_height] .<= (reactor_cells[1:reactor_width,1:reactor_length-1,1:reactor_height] .+ moderators[1:reactor_width,1:reactor_length-1,1:reactor_height]))
    @constraint(model,backward_neutron_flux[1:reactor_width,1:reactor_length-1,1:reactor_height] .- (1 .-moderators[1:reactor_width,1:reactor_length-1,1:reactor_height]) .<= backward_neutron_flux[1:reactor_width,2:reactor_length,1:reactor_height])
    @constraint(model,backward_neutron_flux[1:reactor_width,1:reactor_length-6,1:reactor_height] .<= 4 .+ reactor_cells[1:reactor_width,1:reactor_length-6,1:reactor_height]
    .- moderators[1:reactor_width,2:reactor_length-5,1:reactor_height] .- moderators[1:reactor_width,3:reactor_length-4,1:reactor_height]
    .- moderators[1:reactor_width,4:reactor_length-3,1:reactor_height] .- moderators[1:reactor_width,5:reactor_length-2,1:reactor_height])
    @constraint(model,backward_boosted_reactor[1:reactor_width,reactor_length,1:reactor_height] .== 0)
    @constraint(model,backward_boosted_reactor[1:reactor_width,1:reactor_length-1,1:reactor_height] .<= backward_neutron_flux[1:reactor_width,2:reactor_length,1:reactor_height])
    @constraint(model,backward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[:,1:reactor_length-1,:] .+ reactor_cells[:,2:reactor_length,:] .- 1 .<= backward_boosted_reactor[:,1:reactor_length-1,:])
    @constraint(model,reactor_cells[:,1:reactor_length-2,:] .+ moderators[:,2:reactor_length-1,:] .+ reactor_cells[:,3:reactor_length,:] .- 2 .<= backward_boosted_reactor[:,1:reactor_length-2,:])
    @constraint(model,reactor_cells[:,1:reactor_length-3,:] .+ moderators[:,2:reactor_length-2,:] .+ moderators[:,3:reactor_length-1,:] .+ reactor_cells[:,4:reactor_length,:] .- 3 .<= backward_boosted_reactor[:,1:reactor_length-3,:])
    @constraint(model,reactor_cells[:,1:reactor_length-4,:] .+ moderators[:,2:reactor_length-3,:] .+ moderators[:,3:reactor_length-2,:] .+ moderators[:,4:reactor_length-1,:] .+ reactor_cells[:,5:reactor_length,:] .- 4 .<= backward_boosted_reactor[:,1:reactor_length-4,:])
    @constraint(model,reactor_cells[:,1:reactor_length-5,:] .+ moderators[:,2:reactor_length-4,:] .+ moderators[:,3:reactor_length-3,:] .+ moderators[:,4:reactor_length-2,:] .+ moderators[:,5:reactor_length-1,:] .+ reactor_cells[:,6:reactor_length,:] .- 5 .<= backward_boosted_reactor[:,1:reactor_length-5,:])
    
    #The code style is a bit inconsistent. Artifact of old code. But well, if it works, it works.
    #Leftward.
    @variable(model,leftward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,leftward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,leftward_neutron_flux[1,1:reactor_length,1:reactor_height] .<= reactor_cells[1,1:reactor_length,1:reactor_height])
    @constraint(model,leftward_neutron_flux[2:reactor_width,1:reactor_length,1:reactor_height] .<= (moderators[2:reactor_width,1:reactor_length,1:reactor_height].+reactor_cells[2:reactor_width,1:reactor_length,1:reactor_height]))
    @constraint(model,leftward_neutron_flux[2:reactor_width,1:reactor_length,1:reactor_height] .- (1 .-moderators[2:reactor_width,1:reactor_length,1:reactor_height]) .<= leftward_neutron_flux[1:reactor_width-1,1:reactor_length,1:reactor_height])
    @constraint(model,leftward_neutron_flux[6:reactor_width,1:reactor_length,1:reactor_height] .<= 4 .+ reactor_cells[6:reactor_width,1:reactor_length,1:reactor_height] 
    .- moderators[5:reactor_width-1,1:reactor_length,1:reactor_height] .- moderators[4:reactor_width-2,1:reactor_length,1:reactor_height] 
    .- moderators[3:reactor_width-3,1:reactor_length,1:reactor_height] .- moderators[2:reactor_width-4,1:reactor_length,1:reactor_height])
    @constraint(model,leftward_boosted_reactor[1,1:reactor_length,1:reactor_height] .== 0) #These dummy variables would likely be optimized away. If not, they stay dummy variables without significant impact.
    @constraint(model,leftward_boosted_reactor[2:reactor_width,1:reactor_length,1:reactor_height] .<= leftward_neutron_flux[1:reactor_width-1,1:reactor_length,1:reactor_height])
    @constraint(model,leftward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[2:end,:,:] .+ reactor_cells[1:end-1,:,:] .- 1 .<= leftward_boosted_reactor[2:end,:,:])
    @constraint(model,reactor_cells[3:end,:,:] .+ moderators[2:end-1,:,:] .+ reactor_cells[1:end-2,:,:] .- 2 .<= leftward_boosted_reactor[3:end,:,:])
    @constraint(model,reactor_cells[4:end,:,:] .+ moderators[3:end-1,:,:] .+ moderators[2:end-2,:,:] .+ reactor_cells[1:end-3,:,:] .- 3 .<= leftward_boosted_reactor[4:end,:,:])
    @constraint(model,reactor_cells[5:end,:,:] .+ moderators[4:end-1,:,:] .+ moderators[3:end-2,:,:] .+ moderators[2:end-3,:,:] .+ reactor_cells[1:end-4,:,:] .- 4 .<= leftward_boosted_reactor[5:end,:,:])
    @constraint(model,reactor_cells[6:end,:,:] .+ moderators[5:end-1,:,:] .+ moderators[4:end-2,:,:] .+ moderators[3:end-3,:,:] .+ moderators[2:end-4,:,:] .+ reactor_cells[1:end-5,:,:] .- 5 .<= leftward_boosted_reactor[6:end,:,:])
    
    
    #Rightward.
    @variable(model,rightward_boosted_reactor[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,rightward_neutron_flux[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,rightward_neutron_flux[reactor_width,1:reactor_length,1:reactor_height] .<= reactor_cells[reactor_width,1:reactor_length,1:reactor_height])
    @constraint(model,rightward_neutron_flux[1:reactor_width-1,1:reactor_length,1:reactor_height] .<= (reactor_cells[1:reactor_width-1,1:reactor_length,1:reactor_height] .+ moderators[1:reactor_width-1,1:reactor_length,1:reactor_height]))
    @constraint(model,rightward_neutron_flux[1:reactor_width-1,1:reactor_length,1:reactor_height] .- (1 .-moderators[1:reactor_width-1,1:reactor_length,1:reactor_height]) .<= rightward_neutron_flux[2:reactor_width,1:reactor_length,1:reactor_height])
    @constraint(model,rightward_neutron_flux[1:reactor_width-6,1:reactor_length,1:reactor_height] .<= 4 .+ reactor_cells[1:reactor_width-6,1:reactor_length,1:reactor_height]
    .- moderators[2:reactor_width-5,1:reactor_length,1:reactor_height] .- moderators[3:reactor_width-4,1:reactor_length,1:reactor_height]
    .- moderators[4:reactor_width-3,1:reactor_length,1:reactor_height] .- moderators[5:reactor_width-2,1:reactor_length,1:reactor_height])
    @constraint(model,rightward_boosted_reactor[reactor_width,1:reactor_length,1:reactor_height] .== 0)
    @constraint(model,rightward_boosted_reactor[1:reactor_width-1,1:reactor_length,1:reactor_height] .<= rightward_neutron_flux[2:reactor_width,1:reactor_length,1:reactor_height])
    @constraint(model,rightward_boosted_reactor .<= reactor_cells)
    @constraint(model,reactor_cells[1:reactor_width-1,:,:] .+ reactor_cells[2:reactor_width,:,:] .- 1 .<= rightward_boosted_reactor[1:reactor_width-1,:,:])
    @constraint(model,reactor_cells[1:reactor_width-2,:,:] .+ moderators[2:reactor_width-1,:,:] .+ reactor_cells[3:reactor_width,:,:] .- 2 .<= rightward_boosted_reactor[1:reactor_width-2,:,:])
    @constraint(model,reactor_cells[1:reactor_width-3,:,:] .+ moderators[2:reactor_width-2,:,:] .+ moderators[3:reactor_width-1,:,:] .+ reactor_cells[4:reactor_width,:,:] .- 3 .<= rightward_boosted_reactor[1:reactor_width-3,:,:])
    @constraint(model,reactor_cells[1:reactor_width-4,:,:] .+ moderators[2:reactor_width-3,:,:] .+ moderators[3:reactor_width-2,:,:] .+ moderators[4:reactor_width-1,:,:] .+ reactor_cells[5:reactor_width,:,:] .- 4 .<= rightward_boosted_reactor[1:reactor_width-4,:,:])
    @constraint(model,reactor_cells[1:reactor_width-5,:,:] .+ moderators[2:reactor_width-4,:,:] .+ moderators[3:reactor_width-3,:,:] .+ moderators[4:reactor_width-2,:,:] .+ moderators[5:reactor_width-1,:,:] .+ reactor_cells[6:reactor_width,:,:] .- 5 .<= rightward_boosted_reactor[1:reactor_width-5,:,:])
    

    @variable(model,reactor_heat[1:reactor_width,1:reactor_length,1:reactor_height] >= 0)
    @variable(model,local_heat[1:reactor_width,1:reactor_length,1:reactor_height] >= 0)
    #Moderator heat.
    @variable(model,local_cooling[1:reactor_width,1:reactor_length,1:reactor_height])
    @variable(model,reactor_rf[1:reactor_width,1:reactor_length,1:reactor_height])
    @variable(model,moderator_rf[1:reactor_width,1:reactor_length,1:reactor_height])
    @constraint(model,sum(local_heat)+sum(reactor_heat)-sum(local_cooling) <= 0)
    @objective(model,Max,sum(reactor_rf) + sum(moderator_rf))
    @variable(model, first_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model, second_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model, third_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model, fourth_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model, fifth_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model, sixth_side[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,first_side .<= second_side)
    @constraint(model,second_side .<= third_side)
    @constraint(model,third_side .<= fourth_side)
    @constraint(model,fourth_side .<= fifth_side)
    @constraint(model,fifth_side .<= sixth_side)

    @constraint(model, first_side .+ second_side .+ third_side .+ fourth_side .+ fifth_side .+ sixth_side .>= 
    upward_boosted_reactor .+ downward_boosted_reactor .+ leftward_boosted_reactor .+ rightward_boosted_reactor .+ forward_boosted_reactor .+ backward_boosted_reactor)
    @variable(model,active_moderators[1:reactor_width,1:reactor_length,1:reactor_height],Bin)

    @constraint(model,reactor_rf .<= reactor_cells .+ upward_boosted_reactor .+ downward_boosted_reactor .+ leftward_boosted_reactor .+ rightward_boosted_reactor .+ forward_boosted_reactor .+ backward_boosted_reactor)


    #Now, after dealing with all the fuel cells and moderators, the next would be coolant.
    #I will tackle only passive coolant for now.

    @variable(model,water_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(water_coolants) <= water_limit)
    @variable(model,redstone_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(redstone_coolants) <= redstone_limit)
    @variable(model,quartz_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(quartz_coolants) <= quartz_limit)
    @variable(model,gold_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(gold_coolants) <= gold_limit)
    @variable(model,glowstone_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(glowstone_coolants) <= glowstone_limit)
    @variable(model,lapis_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(lapis_coolants) <= lapis_limit)
    @variable(model,diamond_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(diamond_coolants) <= diamond_limit)
    @variable(model,liquid_helium_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(liquid_helium_coolants) <= liquid_helium_limit)
    @variable(model,enderium_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(enderium_coolants) <= enderium_limit)
    @variable(model,cryotheum_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(cryotheum_coolants) <= cryotheum_limit)
    @variable(model,iron_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(iron_coolants) <= iron_limit)
    @variable(model,emerald_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(emerald_coolants) <= emerald_limit)
    @variable(model,copper_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(copper_coolants) <= copper_limit)
    @variable(model,tin_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(tin_coolants) <= tin_limit)
    @variable(model,magnesium_coolants[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,sum(magnesium_coolants) <= magnesium_limit)

    @constraint(model, local_cooling .<= water_coolants .* water_cooling .+redstone_coolants .* redstone_cooling 
    .+quartz_coolants .* quartz_cooling .+gold_coolants .* gold_cooling .+glowstone_coolants 
    .* glowstone_cooling .+lapis_coolants .* lapis_cooling .+diamond_coolants .* diamond_cooling 
    .+liquid_helium_coolants .* liquid_helium_cooling .+enderium_coolants .* enderium_cooling 
    .+cryotheum_coolants .* cryotheum_cooling .+iron_coolants .* iron_cooling .+emerald_coolants 
    .* emerald_cooling .+copper_coolants .* copper_cooling .+tin_coolants .* tin_cooling 
    .+magnesium_coolants .* magnesium_cooling)
    #Final constraint that all blocks sum up to the maximum of 1 each block. Each block can't be more than one thing at once.

    @constraint(model, reactor_cells .+ moderators .+ water_coolants .+ redstone_coolants .+ quartz_coolants .+ gold_coolants 
    .+ glowstone_coolants .+ lapis_coolants .+ diamond_coolants .+ liquid_helium_coolants .+ enderium_coolants .+ cryotheum_coolants 
    .+ iron_coolants .+ emerald_coolants .+ copper_coolants .+ tin_coolants .+ magnesium_coolants .<= 1)

    #Tin has a special condition that requires extra variables.
    @variable(model,x_axis_tin_suitable[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,y_axis_tin_suitable[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @variable(model,z_axis_tin_suitable[1:reactor_width,1:reactor_length,1:reactor_height],Bin)
    @constraint(model,tin_coolants .<= x_axis_tin_suitable .+ y_axis_tin_suitable .+ z_axis_tin_suitable)
    @constraint(model,moderator_rf .<= 7 .* moderators)

    
    for x in 1:reactor_width, y in 1:reactor_length, z in 1:reactor_height
        neighbors = get_neighbors(x,y,z,reactor_width,reactor_length,reactor_height)

        #These are the heat.

        #Not inactive moderator...
        @constraint(model,active_moderators[x,y,z]*8 <= 7*moderators[x,y,z] + sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))
        @constraint(model,local_heat[x,y,z] >= base_heat*(moderators[x,y,z] - sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))) #Heat for inactive moderators.
        @constraint(model,local_heat[x,y,z] + (56*base_heat)*(1-moderators[x,y,z]) >= sum(reactor_heat[i,j,k] for (i,j,k) in neighbors)/3) #Heat for active moderators
        @constraint(model,reactor_heat[x,y,z] + (28*base_heat)*(1-reactor_cells[x,y,z]) >= base_heat*(1+2*first_side[x,y,z]+3*second_side[x,y,z]+4*third_side[x,y,z]+5*fourth_side[x,y,z]+6*fifth_side[x,y,z]+7*sixth_side[x,y,z]))
        @constraint(model,moderator_rf[x,y,z] <= sum(reactor_rf[i,j,k] for (i,j,k) in neighbors)/6)

        #Now, these are the cooling constraints.

        
        #Water Cooler	60 H/t	Must be adjacent to at least one Reactor Cell or active moderator block.
        @constraint(model,water_coolants[x,y,z] <= sum(reactor_cells[i,j,k] for (i,j,k) in neighbors) + sum(active_moderators[i,j,k] for (i,j,k) in neighbors))
        #Redstone Cooler	90 H/t	Must be adjacent to at least one Reactor Cell.
        @constraint(model,redstone_coolants[x,y,z] <= sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))
        #Quartz Cooler	90 H/t	Must be adjacent to at least one active moderator block.
        @constraint(model,quartz_coolants[x,y,z] <= sum(active_moderators[i,j,k] for (i,j,k) in neighbors))
        #Gold Cooler	120 H/t	Must be adjacent to at least one valid Water Cooler and one valid Redstone Cooler.
        @constraint(model,gold_coolants[x,y,z] <= sum(water_coolants[i,j,k] for (i,j,k) in neighbors))
        @constraint(model,gold_coolants[x,y,z] <= sum(redstone_coolants[i,j,k] for (i,j,k) in neighbors))
        #Glowstone Cooler	130 H/t	Must be adjacent to at least two active moderator blocks.
        @constraint(model,2*glowstone_coolants[x,y,z] <= sum(active_moderators[i,j,k] for (i,j,k) in neighbors))
        #Lapis Cooler	120 H/t	Must be adjacent to at least one Reactor Cell and one Reactor Casing.
        if length(neighbors) <= 5
            @constraint(model,lapis_coolants[x,y,z] <= sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))
        else
            @constraint(model,lapis_coolants[x,y,z] <= 0)
        end
        #Diamond Cooler	150 H/t	Must be adjacent to at least one valid Water Cooler and one valid Quartz Cooler.
        @constraint(model,diamond_coolants[x,y,z] <= sum(water_coolants[i,j,k] for (i,j,k) in neighbors))
        @constraint(model,diamond_coolants[x,y,z] <= sum(quartz_coolants[i,j,k] for (i,j,k) in neighbors))
        #Liquid Helium Cooler	140 H/t	Must be adjacent to exactly one valid Redstone Cooler and at least one Reactor Casing.
        if length(neighbors) <= 5
            @constraint(model,liquid_helium_coolants[x,y,z] <= sum(redstone_coolants[i,j,k] for (i,j,k) in neighbors)) #Must have at least one redstone coolant.
            @constraint(model,liquid_helium_coolants[x,y,z]*6 + sum(redstone_coolants[i,j,k] for (i,j,k) in neighbors) <= 7) #Cannot exceed one.
        else
            @constraint(model,liquid_helium_coolants[x,y,z] <= 0)
        end
        #Enderium Cooler*	120 H/t	Must be adjacent to exactly three Reactor Casings at exactly one vertex.
        if length(neighbors) != 3
            @constraint(model,enderium_coolants[x,y,z] <= 0)
        end
        #Cryotheum Cooler*	160 H/t	Must be adjacent to at least two Reactor Cells.
        @constraint(model,2*cryotheum_coolants[x,y,z] <= sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))
        #Iron Cooler	80 H/t	Must be adjacent to at least one valid Gold Cooler.
        @constraint(model,iron_coolants[x,y,z] <= sum(gold_coolants[i,j,k] for (i,j,k) in neighbors))
        #Emerald Cooler	160 H/t	Must be adjacent to at least one active moderator block and one Reactor Cell.
        @constraint(model,emerald_coolants[x,y,z] <= sum(reactor_cells[i,j,k] for (i,j,k) in neighbors))
        @constraint(model,emerald_coolants[x,y,z] <= sum(active_moderators[i,j,k] for (i,j,k) in neighbors))
        #Copper Cooler	80 H/t	Must be adjacent to at least one valid Glowstone Cooler.
        @constraint(model,copper_coolants[x,y,z] <= sum(glowstone_coolants[i,j,k] for (i,j,k) in neighbors))
        #Tin Cooler	120 H/t	Must be at least between two valid Lapis Coolers along the same axis.
        if ((x-1,y,z) in neighbors) & ((x+1,y,z) in neighbors)
            @constraint(model,2*x_axis_tin_suitable[x,y,z] <= lapis_coolants[x-1,y,z] + lapis_coolants[x+1,y,z])
        else
            @constraint(model,x_axis_tin_suitable[x,y,z]<=0)
        end

        if ((x,y-1,z) in neighbors) & ((x,y+1,z) in neighbors)
            @constraint(model,2*y_axis_tin_suitable[x,y,z] <= lapis_coolants[x,y-1,z] + lapis_coolants[x,y+1,z])
        else
            @constraint(model,y_axis_tin_suitable[x,y,z]<=0)
        end

        if ((x,y,z-1) in neighbors) & ((x,y,z+1) in neighbors)
            @constraint(model,2*z_axis_tin_suitable[x,y,z] <= lapis_coolants[x,y,z-1] + lapis_coolants[x,y,z+1])
        else
            @constraint(model,z_axis_tin_suitable[x,y,z]<=0)
        end
        #The final constraint that sums the suitability was given above
        #Magnesium Cooler	110 H/t	Must be adjacent to at least one Reactor Casing and one active moderator block.
        if length(neighbors) <= 5
            @constraint(model,magnesium_coolants[x,y,z] <= sum(active_moderators[i,j,k] for (i,j,k) in neighbors))
        else
            @constraint(model,magnesium_coolants[x,y,z] <= 0)
        end
    end
    
    #println(model)

    

    optimize!(model)
    
    #=
    println(objective_value(model)*base_energy)
    println(value.(reactor_cells))
    println(value.(reactor_heat))
    println(value.(first_side))
    println(value.(upward_boosted_reactor))
    println(value.(local_heat))
    println(value.(local_cooling))
    println(value.(cryotheum_coolants))
    println(value.(emerald_coolants))
    =#

    #=
    println("reactor_rf: ",base_energy .* value.(reactor_rf))
    println("moderator_rf: ",base_energy .* value.(moderator_rf))
    println("moderator_heat: ", value.(local_heat))
    println("reactor_heat: ", value.(reactor_heat))
    println("sixth_side",value.(sixth_side))

    println(value.(downward_boosted_reactor))
    println(value.(upward_boosted_reactor))
    println(value.(forward_boosted_reactor))
    println(value.(backward_boosted_reactor))
    println(value.(leftward_boosted_reactor))
    println(value.(rightward_boosted_reactor))
    =#
    println("power: ",round(objective_value(model)*base_energy,digits=4))
    println("heat: ",round(sum(value.(reactor_heat))+sum(value.(local_heat)),digits=4))
    println("cooling: ",round(sum(value.(local_cooling)),digits=4))

    
    for z in 1:reactor_height
        for y in 1:reactor_length
            for x in 1:reactor_width
                if value(reactor_cells[x,y,z] )≈ 1.0
                    print("[]")
                elseif value(moderators[x,y,z]) ≈ 1.0
                    print("##")
                elseif value(water_coolants[x,y,z]) ≈ 1.0
                    print("Wt")
                elseif value(redstone_coolants[x,y,z]) ≈ 1.0
                    print("Rs")
                elseif value(quartz_coolants[x,y,z]) ≈ 1.0
                    print("Qz")
                elseif value(gold_coolants[x,y,z]) ≈ 1.0
                    print("Au")
                elseif value(glowstone_coolants[x,y,z]) ≈ 1.0
                    print("Gs")
                elseif value(lapis_coolants[x,y,z]) ≈ 1.0
                    print("Lp")
                elseif value(diamond_coolants[x,y,z]) ≈ 1.0
                    print("Dm")
                elseif value(liquid_helium_coolants[x,y,z]) ≈ 1.0
                    print("He")
                elseif value(enderium_coolants[x,y,z]) ≈ 1.0
                    print("Ed")
                elseif value(cryotheum_coolants[x,y,z]) ≈ 1.0
                    print("Cr")
                elseif value(iron_coolants[x,y,z]) ≈ 1.0
                    print("Fe")
                elseif value(emerald_coolants[x,y,z]) ≈ 1.0
                    print("Em")
                elseif value(copper_coolants[x,y,z]) ≈ 1.0
                    print("Cu")
                elseif value(tin_coolants[x,y,z]) ≈ 1.0
                    print("Sn")
                elseif value(magnesium_coolants[x,y,z]) ≈ 1.0
                    print("Mg") 
                else
                    print("  ")
                end
            end
            println()
        end
        println("---------------------------------------------")
    end
end




nuclearcraftoptimize(672,375,3,3,3,num_threads = 8,time_limit = 300.0)