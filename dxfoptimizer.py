
import shapely.affinity
import argparse
import logging
import dxfgrabber
import random
import numpy
import multiprocessing

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from matplotlib import pyplot
from shapely.geometry import Polygon, MultiPolygon
from descartes.patch import PolygonPatch

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
    
def get_extent(polyline):
    extents = [None, None, None, None]
    
    for i in range(2):
        for point in polyline.points:
            if extents[i*2] is None or extents[i*2] > point[i]:
                extents[i*2] = point[i]
            if extents[i*2+1] is None or extents[i*2+1] < point[i]:
                extents[i*2+1] = point[i]
    
    return extents
       
def strip_excess_coords(points):
    return map(lambda p: p[:2], points)
    
def scale(points, scale):
    return [[s*v for s,v in zip(scale,point)] for point in points]

def shift(points, offset):
    return [[v+o for v,o in zip(point, offset)] for point in points]

def mapfuncwithargs(func, iterable, *args):
    return map(lambda v: func(v, *args), iterable)
    
def apply_offsets(individual, shapes):
    offsets = []
    angles = []

    for i in range(len(individual)/3):
        offsets.append((individual[i*3],individual[i*3+1]))
        angles.append(individual[i*3+2])
        
    offset_polygons = []
    
    for angle, offset, polygon in zip(angles, offsets, shapes):
        offset_polygon = shapely.affinity.translate(polygon, xoff=offset[0], yoff=offset[1])
        offset_polygon = shapely.affinity.rotate(offset_polygon, angle, use_radians=True)
        offset_polygons.append(offset_polygon)
        
    return offset_polygons   
    

def draw_intersections(ax, polygons):
    for polygon1 in polygons:
        for polygon2 in polygons:
            if polygon1 == polygon2:
                continue
                    
            intersection = polygon1.intersection(polygon2)
            
            if intersection.geom_type == 'Polygon':
                patch = PolygonPatch(intersection, facecolor=(1.0,0.,0.), edgecolor=v_color(intersection), alpha=0.5, zorder=2)
                ax.add_patch(patch) 
                
def evaluate(individual):
    global polygons
    offsets = []
    angles = []
    
    for i in range(len(individual)/3):
        offsets.append((individual[i*3],individual[i*3+1]))
        angles.append(individual[i*3+2])
        
    offset_polygons = []
    
    for angle, offset, polygon in zip(angles, offsets, polygons):
        offset_polygon = shapely.affinity.translate(polygon, xoff=offset[0], yoff=offset[1])
        offset_polygon = shapely.affinity.rotate(offset_polygon, angle, use_radians=True)
        offset_polygons.append(offset_polygon)
        
    fitness = 0.
        
    intersection_area = 0.
    for polygon1 in offset_polygons:
        for polygon2 in offset_polygons:
            if polygon1 == polygon2:
                continue
                
            intersection_area += polygon1.intersection(polygon2).area
    
    all_objects = MultiPolygon(offset_polygons)
    convex_hull = all_objects.convex_hull
    area = Polygon(convex_hull.exterior.coords, map(lambda p: p.exterior.coords, offset_polygons)).area
        
    fitness = intersection_area*10+area
    
    return fitness,
    
def draw_individual(ax, individual):
    if ax is None:
        return 
        
    ax.cla()
    offset_polygons = apply_offsets(individual, polygons)
    
    for offset_polygon in offset_polygons:
        plot_coords(ax, offset_polygon.exterior)

        patch = PolygonPatch(offset_polygon, facecolor=v_color(offset_polygon), edgecolor=v_color(offset_polygon), alpha=0.5, zorder=1)
        ax.add_patch(patch)
    
        outline = MultiPolygon(offset_polygons).convex_hull
        patch = PolygonPatch(outline, facecolor=(.0,1.,0.), alpha=0.2, zorder=0)
        ax.add_patch(patch) 
        
        draw_intersections(ax, offset_polygons)
        
    pyplot.draw()
    pyplot.show()
        
    
    from time import sleep
    
    #sleep(0.1)

def create_toolbox(num_shapes):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_offset", random.uniform, 0, 5.)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_offset, n=num_shapes*3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #pool = multiprocessing.Pool()
    #toolbox.register("map", pool.map)
        
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1., indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=2)
    
    def evaluate_invalid(individuals):
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    toolbox.register("evaluate_invalid", evaluate_invalid)
    
    def greedy_opt(individual, visualize=False, ax=None):
        
        stepsize = 1.
        for i in range(len(individual)):
            
            while stepsize > 0.05:
                
                ind1 = toolbox.clone(individual)
                ind1[i] += stepsize
                ind1.fitness.values = toolbox.evaluate(ind1)
                
                ind2 = toolbox.clone(individual)
                ind2[i] -= stepsize
                ind2.fitness.values = toolbox.evaluate(ind2)
                
                if ind1.fitness.values < individual.fitness.values:
                    individual = ind1
                    individual.fitness.values = ind1.fitness.values
                    continue
                    
                if ind2.fitness.values < individual.fitness.values:
                    individual = ind2
                    individual.fitness.values = ind2.fitness.values
                    continue
                    
                stepsize = stepsize/2.
                
                #print "No improvement"
                draw_individual(ax, individual)
            
            stepsize = 1.
            
            
            #print "Stepsize", i, stepsize
            
                
    
    toolbox.register("greedy_opt", greedy_opt)
    
    return toolbox

def parse_shapes(dxf_files):
    
    dxf = dxfgrabber.readfile("dummy_part.dxf")
    print("DXF version: {}".format(dxf.dxfversion))
    header_var_count = len(dxf.header) # dict of dxf header vars
    layer_count = len(dxf.layers) # collection of layer definitions
    block_definition_count = len(dxf.blocks) #  dict like collection of block definitions
    entity_count = len(dxf.entities) # list like collection of entities
    print header_var_count, layer_count, block_definition_count, entity_count


    all_lines = [entity for entity in dxf.entities if entity.dxftype == 'SPLINE']
    all_entities_at_layer_0 = [entity for entity in dxf.entities if entity.layer == '0']

    
    
    polygons = []


    for polyline in dxf.entities:
        if entity.dxftype != "POLYLINE":
            continue
     
        points = polyline.points

        polygon = Polygon(map(tuple,points))
        
        #plot_coords(ax, polygon.interiors)

        polygons.append(polygon)

    
    return polygons

polygons = object()

    

def optimize(toolbox, shapes, visualize=False):
    global polygons
    
    polygons = shapes
    
    if visualize:
        fig = pyplot.figure(1,  dpi=90)
        ax = fig.gca()

        pyplot.axis('equal')
        pyplot.ion()
        pyplot.show()
        
    random.seed(64)
    
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
        
    pop = toolbox.population(n=100)
    CXPB, MUTPB, NGEN = 0.2, 0.5, 100

    # Evaluate the entire population
    toolbox.evaluate_invalid(pop)
    
    while True:
        for g in range(NGEN):
            print "Evolvign"
            
            # Select the next generation individuals
            
            offspring = toolbox.select(pop, 2*len(pop))
            # Clone the selected individuals
            offspring = map(toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            toolbox.evaluate_invalid(offspring)

            # The population is entirely replaced by the offspring
            
            
            mixed_pop = toolbox.population(n=3*len(pop))
            mixed_pop[:len(pop)] = pop[:]
            mixed_pop[len(pop):] = offspring
            
            #print map(lambda i: i.fitness.values, mixed_pop), len(mixed_pop)
            
            new_generation = tools.selBest(mixed_pop, len(pop))
            pop[:] = new_generation



            #pop[:] = toolbox.map(toolbox.greedy_opt, pop)
            
            print "Generation", g, stats.compile(pop)
            
            
            print "Greedy optimization"
            
            individual = tools.selBest(pop,1)[0]
            
            oldfit = individual.fitness.values
            toolbox.greedy_opt(individual, visualize=True, ax=ax)
            
            print oldfit, individual.fitness.values
            #if visualize:
                
                
                
                #ax.cla()
                #offset_polygons = apply_offsets(individual, shapes)
            
                #for offset_polygon in offset_polygons:
                    #plot_coords(ax, offset_polygon.exterior)

                    #patch = PolygonPatch(offset_polygon, facecolor=v_color(offset_polygon), edgecolor=v_color(offset_polygon), alpha=0.5, zorder=1)
                    #ax.add_patch(patch)
                
                #outline = MultiPolygon(offset_polygons).convex_hull
                #patch = PolygonPatch(outline, facecolor=(.0,1.,0.), alpha=0.2, zorder=0)
                #ax.add_patch(patch) 
                
                #draw_intersections(ax, offset_polygons)
                
                #pyplot.draw()
                #pyplot.show()


        #Nuke population
        
        for mutant in pop:
            for i in range(len(shapes)*2):
                toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        toolbox.evaluate_invalid(pop)
            
        print "Nuked"

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf_file", type=str, help="One or more dxf files to optimizie", nargs="+")

    parser.add_argument("--exclusion_radius", type=float, default=0, help="Exclusion radius")

    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s:%(levelname)s:%(message)s")

    parser = create_parser()
    args = parser.parse_args()

    shapes = parse_shapes([])
    toolbox = create_toolbox(len(shapes))
    
    optimize(toolbox, shapes, visualize=args.display)

