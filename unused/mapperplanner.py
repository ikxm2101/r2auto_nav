class MapperPlaner(Node):
    '''
    Constructor
    '''

    def __init__(self, name: str = 'mapperplanner'):
        super().__init__(name)
        self.params = MapperPlanerParameters()
        self.init_topics()

    '''
    Costmap
    '''

    def flatten_layers(self):
        self.params.inflation_layer = self.params.inflation_layer.flatten()
        self.params.obstacle_layer = self.params.obstacle_layer.flatten()

    def reshape_layers(self):
        if self.params.inflation_layer.any():
            zoom_y = self.params.height / self.params.prev_height
            zoom_x = self.params.width / self.params.prev_width

            # Use scipy's zoom function to resize the array
            self.params.inflation_layer = scipy.ndimage.zoom(
                self.params.inflation_layer, (zoom_y, zoom_x))

            # Ensure the values are still in the correct range after interpolation
            self.params.inflation_layer = np.clip(
                self.params.inflation_layer, self.params.inflation_min, self.params.inflation_max)
            self.params.inflation_layer = np.uint8(self.params.inflation_layer)
            # self.params.inflation_layer = np.uint8(self.params.inflation_layer.reshape(self.params.height, self.params.width)) # occ_y-major

        if self.params.obstacle_layer.any():
            zoom_y = self.params.height / self.params.prev_height
            zoom_x = self.params.width / self.params.prev_width

            # Use scipy's zoom function to resize the array
            self.params.obstacle_layer = scipy.ndimage.zoom(
                self.params.obstacle_layer, (zoom_y, zoom_x))

            # Ensure the values are still in the correct range after interpolation
            self.params.obstacle_layer = np.clip(
                self.params.obstacle_layer, self.params.inflation_min, self.params.inflation_max)
            self.params.obstacle_layer = np.uint8(self.params.obstacle_layer)
            # self.params.obstacle_layer = np.uint8(self.params.obstacle_layer.reshape(self.params.height, self.params.width)) # occ_y-major

    def update_inflationlayer(self):
        # initialise inflation layer with cost of zeroes
        self.params.inflation_layer = np.zeros(
            (self.params.height, self.params.width))

        radius = np.ceil((self.params.inflation_radius /
                         self.params.resolution))  # in grid cells

        # Get the indices of the occupied cells
        occupied_indices = np.where(self.occ_2d == self.params.occupied)
        # self.get_logger().info(f'occupied indices: {occupied_indices}')

        for y, x in zip(*occupied_indices):
            # Range of cells to inflate around the occupied cell
            x_min = int(max(0, x - radius))
            x_max = int(min(self.params.width, x + radius + 1))
            y_min = int(max(0, y - radius))
            y_max = int(min(self.params.height, y + radius + 1))

            # Inflate the cells
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    # Calculate the distance from the occupied cell
                    dist = np.sqrt((i - y)**2 + (j - x)**2)
                    # Calculate the cost to be added to the cell
                    if dist <= radius:
                        # if distance is within the inflation radius
                        cost = self.params.inflation_inc
                        # Update the cost of the cell
                        self.params.inflation_layer[i, j] = min(
                            self.params.inflation_max, self.params.inflation_layer[i, j] + cost)

        # Publishing inflation layer
        layer_msg = OccupancyGrid()
        now = rclpy.time.Time()
        # Set the header
        layer_msg.header.frame_id = 'map'
        layer_msg.header.stamp = now.to_msg()
        # Set the info
        layer_msg.info.resolution = self.params.resolution
        layer_msg.info.width = self.params.width
        layer_msg.info.height = self.params.height
        layer_msg.info.origin.position.x = self.params.origin_x
        layer_msg.info.origin.position.y = self.params.origin_y
        # Flatten the layer and convert it to a list of integers
        # self.get_logger().info(f'{min(self.params.inflation_layer.flatten())}, {max(self.params.inflation_layer.flatten())}')
        layer_msg.data = list(map(int, self.params.inflation_layer.flatten()))

        # Publish the inflation layer message
        self.inflation_publisher.publish(layer_msg)

    def update_obstaclelayer(self):
        # initialiise obstacle layer by creating a copy of map_2d
        self.params.obstacle_layer = self.occ_2d.copy()
        # assign costs to cells accordingly
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.unknown] = self.params.obstacle_unknown
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.unoccupied] = self.params.obstacle_min
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.occupied] = self.params.obstacle_max

        # Publishing obstacle layer
        layer_msg = OccupancyGrid()
        now = rclpy.time.Time()
        # Set the header
        layer_msg.header.frame_id = 'map'
        # layer_msg.header.stamp = now.to_msg()
        # Set the info
        layer_msg.info.resolution = self.params.resolution
        layer_msg.info.width = self.params.width
        layer_msg.info.height = self.params.height
        layer_msg.info.origin.position.x = self.params.origin_x
        layer_msg.info.origin.position.y = self.params.origin_y
        # Flatten the layer and convert it to a list of integers
        self.get_logger().info(
            f'{min(self.params.obstacle_layer.flatten())}, {max(self.params.obstacle_layer.flatten())}')
        layer_msg.data = list(map(int, self.params.obstacle_layer.flatten()))

        # Publish the obstacle layer message
        self.obstacle_publisher.publish(layer_msg)

    def create_costmap(self, src: np.ndarray, dilate=2, threshold=40, inflation_radius=4, inflation_step=2, erode=2):
        array_eroded: np.ndarray = src.copy()
        array_eroded[src == -1] = 1
        array_eroded[src != -1] = 0
        array_eroded = cv.morphologyEx(
            np.uint8(array_eroded),
            cv.MORPH_CLOSE,
            cv.getStructuringElement(
                cv.MORPH_RECT, (2 * erode + 1, 2 * erode + 1))
        )  # Dilate then erode the unknown points, to get rid of stray wall and stray unoccupied points

        # create a copy of the occ_2d consisting of values, -1 to 100 indicating probability of occupancy
        array_dilated: np.ndarray = src.copy()
        # any cell with probability of possibly being occupied is assigned an inflation step
        array_dilated[src >= threshold] = inflation_step
        # any cell with probability of not being occupied is assigned 0
        array_dilated[src < threshold] = 0
        array_dilated = cv.dilate(
            np.uint8(array_dilated),
            cv.getStructuringElement(
                cv.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1)),
        )

        wall_indexes = np.nonzero(array_dilated)

        array_inflated: np.ndarray = array_dilated.copy()

        for _ in range(inflation_radius):
            array_inflated = array_inflated + \
                cv.dilate(np.uint8(array_dilated),
                          cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
            array_dilated = array_inflated.copy()
            array_dilated[array_dilated != 0] = inflation_step

        array_inflated = np.int8(array_inflated)
        array_inflated[array_eroded == 1] = -1
        array_inflated[wall_indexes] = 100
        return array_inflated

    def update_costmap(self):
        raw_occ_2d = self.occ_data.reshape(
            self.params.height, self.params.width)
        self.params.costmap = self.create_costmap(
            raw_occ_2d,
            # the higher this is, the further i avoid straight walls
            dilate=int((0.273 / 2) // self.params.resolution + 1),
            inflation_radius=4,
            inflation_step=24,
            threshold=52,
            erode=6)

        # create costmap message
        # now = rclpy.time.Time()
        # costmap_msg = OccupancyGrid()
        # # header
        # costmap_msg.header.frame_id = 'map'
        # costmap_msg.header.stamp = now.to_msg()
        # # info
        # costmap_msg.info.resolution = self.params.resolution
        # costmap_msg.info.height = self.params.height
        # costmap_msg.info.width = self.params.width
        # costmap_msg.info.origin.position.x = self.params.origin_x
        # costmap_msg.info.origin.position.y = self.params.origin_y
        # # data
        # costmap_msg.data = Array(
        #     'b', self.params.costmap.ravel().astype(np.int8))

        # # add the costs on the inflation layer
        # self.params.costmap = self.params.inflation_layer + self.params.obstacle_layer
        # # Clip the costs to the range 0 to 127
        # self.params.costmap = np.clip(self.params.costmap, 0, 127)

        # ## data
        # # self.get_logger().info(f'{min(self.params.costmap.flatten())}, {max(self.params.costmap.flatten())}')
        # costmap_msg.data = Array(map(int, self.params.costmap.flatten())) # Flatten the costmap and convert it to a list of integers

        # Publish the costmap message
        # self.costmap_publisher.publish(costmap_msg)