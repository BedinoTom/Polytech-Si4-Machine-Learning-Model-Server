def is_point(x, y, coord):
    if coord[0] <= x and coord[1] <= y and coord[2] >= x and coord[3] >= y:
            return True
    return False

class Square:
    def __init__(self, label, points, coords):
        self.label = label
        self.points = points
        self.coords = coords
        
    def is_point(self, x, y):
        for i in self.points:
            if i[0] == x and i[1] == y:
                return True
        return False
    
    def is_valide_point(self, label, x, y):
        if self.is_point(x, y):
            return False
        
        if label != self.label:
            return False
        
        if self.coords[0] <= x and self.coords[1] <= y and self.coords[2] >= x and self.coords[3] >= y:
            return True
        return False
    
    def add_point(self, x, y):
        self.points.append([x,y])
        
    def grow(self, points, classes, x, y, label):
        new_coord = [
            self.coords[0],
            self.coords[1],
            self.coords[2],
            self.coords[3]
        ]
        if x < self.coords[0] and y < self.coords[1]:
            new_coord = [
                x,
                y,
                self.coords[2],
                self.coords[3]
            ]
        elif x < self.coords[0]:
            new_coord = [
                x,
                self.coords[1],
                self.coords[2],
                self.coords[3]
            ]
        elif y < self.coords[1]:
            new_coord = [
                self.coords[0],
                y,
                self.coords[2],
                self.coords[3]
            ]
        elif x > self.coords[2] and y > self.coords[3]:
            new_coord = [
                self.coords[0],
                self.coords[1],
                x,
                y
            ]
        elif x > self.coords[2]:
            new_coord = [
                self.coords[0],
                self.coords[1],
                x,
                self.coords[3]
            ]
        elif y > self.coords[3]:
            new_coord = [
                self.coords[0],
                self.coords[1],
                self.coords[2],
                y
            ]
        for point in range(0,len(points)):
            reduce_point = points[point][0:2]
            if classes[point] != label:
                if is_point(reduce_point[0], reduce_point[1], new_coord):
                    return False
        self.coords = new_coord
        return True

class SquareClassifier:
    def __init__(self, size=10):
        self.squares = []
        self.size = size
    
    def fit(self, x, y):
        for point in range(0,len(x)):
            reduce_point = x[point][0:2]
            new_squares=[]
            if len(self.squares) == 0:
                coord = [
                            reduce_point[0]-self.size,
                            reduce_point[1]-self.size,
                            reduce_point[0]+self.size,
                            reduce_point[1]+self.size
                        ]
                sq = Square(y[point], [reduce_point], coord)
                self.squares.append(sq)
            process = False
            for square in self.squares:
                if square.is_valide_point(y[point], reduce_point[0], reduce_point[1]):
                    square.add_point(reduce_point[0], reduce_point[1])
                    process = True
                    break
                elif square.label == y[point]:
                    if not square.grow(x, y, reduce_point[0], reduce_point[1], y[point]):
                        coord = [
                            reduce_point[0]-self.size,
                            reduce_point[1]-self.size,
                            reduce_point[0]+self.size,
                            reduce_point[1]+self.size
                        ]
                        sq = Square(y[point], [reduce_point], coord)
                        new_squares.append(sq)
                        process = True
                        break
            if not process:
                coord = [
                            reduce_point[0]-self.size,
                            reduce_point[1]-self.size,
                            reduce_point[0]+self.size,
                            reduce_point[1]+self.size
                        ]
                sq = Square(y[point], [reduce_point], coord)
                self.squares.append(sq)
            self.squares.extend(new_squares)
    
    def predict(self, x):
        res = []
        for point in range(0,len(x)):
            reduce_point = x[point][0:2]
            f = False
            for square in self.squares:
                if is_point(reduce_point[0], reduce_point[1], square.coords):
                    res.append(square.label)
                    f = True
                    break
            if not f:
                res.append(-1)
        return res
    
    def score(self, x, y):
        y_pred = self.predict(x)
        valid = 0
        for i in range(0,len(y_pred)):
            if y_pred[i] == y[i]:
                valid = valid + 1
        return (valid*100)/len(y)