import numpy as np
import matplotlib.pyplot as plt


def calc_fib_coords(N, R=0.2):
    """A function to generate a sequence of fibre pivot positions that are equally spaced

      Returns:
          _type_: an array of XY values with a fibre number
      """
    assert type(N)==int, 'Number of positions must be an integer'
      
    R += 1
    coordinates_cartesian = []
    coordinates_radial = []
    ids = np.arange(N)
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        ID = ids[i]
        coordinates_cartesian.append((ID, x, y))
        coordinates_radial.append((ID, R, theta))  
      
    return np.asarray(coordinates_cartesian), np.asarray(coordinates_radial)

def calc_fibre_dist(fibre_start, fibre_end):
    '''A function to calculate the distance for a single fibre'''
    try:
        distances = []
        for start, end in zip(fibre_start, fibre_end):
            dist = np.linalg.norm(end - start)
            #dist = np.sqrt(distance_vect[0]**2 + distance_vect[1]**2)

            distances.append(dist)
        return np.sum(distances)
        
    except:
        ValueError('Could not evaluate minimum distance')
    
     

 

def generate_points(coords, angle_range_degrees):
    """Generate random points within cones around each coordinate point.

    Args:
        coords (array): An array of XY coordinates.
        angle_range_degrees (float): The angle range of the cone in degrees.

    Returns:
        array: An array of generated points.
    """
    targets_cartesian = []
    targets_radial = []
    for item in coords:
        # Extract coordinates and ID
        Rin, theta = item
        x = Rin*np.cos(theta)
        y = Rin*np.sin(theta)

        # Generate random angle within cone range
        rand_ang = np.radians(np.random.uniform(-angle_range_degrees, angle_range_degrees))

        # Generate random radius within cone radius
        rand_radius = 1.2-np.sqrt(np.random.uniform(0., 1.))
        
        x_shift = rand_radius * np.cos(rand_ang)
        y_shift = rand_radius * np.sin(rand_ang)
        
        rotate = np.array([[np.cos(-theta), -np.sin(-theta)],
                       [np.sin(-theta), np.cos(-theta)]])
        
        x_shift,y_shift = np.dot(rotate,np.array([x_shift,y_shift]))

        # Calculate coordinates of new point
        new_x  = x - x_shift
        new_y = y + y_shift
        
        new_R = np.sqrt(new_x**2. + new_y**2.)
        new_theta = np.arctan(y/x)
        
        if new_x < 0:
            new_theta = np.pi - new_theta
        

        targets_cartesian.append((new_x, new_y))
        targets_radial.append((new_R, new_theta))
    return np.asarray(targets_cartesian), np.asarray(targets_radial)



def alternate_vectors(array1, array2):
    S = []
    for a1, a2 in zip(array1, array2):
        S.append(a1)
        S.append(a2)
    return np.array(S).flatten()


def plot_field(start_coords, end_coords):
    fig, ax = plt.subplots()

    # Create theta values from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, 100)

    # Define x and y coordinates of the unit circle
    x = 1.2*np.cos(theta)
    y = 1.2*np.sin(theta)

    # Plot the unit circle
    ax.plot(x, y)
    
    start_coords = start_coords.reshape(-1, 2)
    end_coords = end_coords.reshape(-1, 2)
    
    plt.scatter(start_coords[:, 0], start_coords[:, 1], color='black', s=30)
    plt.scatter(end_coords[:,0], end_coords[:, 1], color='red', s=30)

    # Set aspect ratio to equal to get a proper circle
    ax.set_aspect('equal', 'box')

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Unit Circle')
    
    plt.show()
    

class field_gen():
    def __init__(self, total_fibres):
        
        start_coords, radial_start = calc_fib_coords(total_fibres)
        
        self.fibre_positions_origin = start_coords[:, 1:]
        self.fibre_positions_origin_radial = radial_start[:, 1:]
        
        self.ids = self.fibre_positions_origin[:, 0].astype(int)
        self.fibre_end_positions, self.fibre_ends_radial = generate_points(self.fibre_positions_origin_radial, 14)
        
        self.flattened_coords_cartesian = alternate_vectors(self.fibre_positions_origin, self.fibre_end_positions)
        self.flattened_coords_radial = alternate_vectors(self.fibre_positions_origin_radial, self.fibre_ends_radial)
        
        self.minimum_distance = calc_fibre_dist(self.fibre_positions_origin, self.fibre_end_positions)
            
        return
    
    def field_img(self):
        plot_field(self.fibre_positions_origin, self.fibre_end_positions)
        return
        
if __name__ == '__main__':
    fg = field_gen(100)
    
    #fg.field_img()
    test_pos = np.empty((1000,2))
    print(test_pos[0])
    for i in range(1000):
        cart, rad = generate_points(np.array([[1.2,0. ]]), 14)
        test_pos[i] = cart
    
    #fibs_cart, fibs_rad = calc_fib_coords(30, R=0.2)
    #print(fibs_rad)
    #fig, ax = plt.subplots()

    # # Create theta values from 0 to 2*pi
    # theta = np.linspace(0, 2*np.pi, 100)

    # # Define x and y coordinates of the unit circle
    # x = 1.2*np.cos(theta)
    # y = 1.2*np.sin(theta)

    # # Plot the unit circle
    # ax.plot(x, y)
    
    # plt.scatter(test_pos[:, 0], test_pos[:, 1])
    # plt.show()
    # ends = fg.fibre_ends_radial
    
    
    
    
    
    #print(ends)
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='polar')
    #c = ax.scatter(fibs_rad[:, 2], fibs_rad[:, 1])
    #plt.show()
    #
    #
    #plt.show()


    
    